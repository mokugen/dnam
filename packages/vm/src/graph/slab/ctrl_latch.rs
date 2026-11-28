//! `3 kHz` control-bus target latching, gauge-invariant `z_stage` `Op` modulation laws, and
//! `#[inline(never)]` L1 register hoisting (`#![no_std]`, `#![forbid(unsafe_code)]`).

#![forbid(unsafe_code)]

use core::f32::consts::PI;

use dnam_kernels::algebraic_unit_rotor_f32;

use crate::control::bus::laws::{
    eval_block_and_group_adaa_lru, eval_global_stage_mods, eval_op_n_pre_modulated,
    eval_phasor_inc_core_multirate_q32, eval_phasor_inc_vm_q32, eval_transport_dry_mix_stage,
    eval_transport_target_psi,
};
use crate::graph::GraphBackend;
use crate::graph::convolver::kautz::{MAX_KAUTZ_BANKS, retune_kautz_banks_3khz};
use crate::graph::transport::{TRANSPORT_MAX_QUADS, TransportModSlots, latch_quad_nonslew_targets};
use crate::math::f32_to_fp16;
use crate::types::Tier;

use super::hydrate::apply_tier::compute_op_n_pre_hoisted;
use super::layout::SonaHotSlab;

const PSI_SLEW_MAX_RAD: f32 = 0.25;

/// Always-inlined `hoist_core_quads_and_readouts` helper for `#[target_feature]` block dispatch.
#[inline(always)]
pub(crate) fn hoist_core_quads_and_readouts_inline<B: GraphBackend>(slab: &mut SonaHotSlab<'_, B>) {
    let t = slab.cold().latched_stage_params;
    let z = slab.cold().latched_z_stage;
    let is_ultra = matches!(slab.lite.tier, Tier::Ultra);

    let (base_op_n, rho_vm) = (slab.cold().base_op_n_pre, slab.cold().zoh.rho_vm as f32);
    let (gamma_bc, zoh_hold, beta_q) = eval_op_n_pre_modulated(
        base_op_n.gamma_bc.get(),
        base_op_n.zoh_hold_depth.get(),
        crate::math::fp16_to_f32(base_op_n.beta_q_fp16.get()),
        &t,
    );
    let hoisted = compute_op_n_pre_hoisted(gamma_bc, zoh_hold, beta_q, rho_vm);
    let tank_dry_mix_eff = eval_transport_dry_mix_stage(slab.cold().base_tank_dry_mix, t[22]);
    slab.lite.tank_dry_mix_eff_fp16 = f32_to_fp16(tank_dry_mix_eff);

    let (fs_vm, inv_fs_vm, base_f, m_core) = (
        slab.cold().zoh.f_s_vm as f32,
        slab.cold().zoh.inv_f_s_vm(),
        slab.cold().base_phasor_freq_hz,
        slab.cold().zoh.m_core,
    );

    {
        let cold = slab.cold_mut();
        eval_block_and_group_adaa_lru(
            (&t, &z),
            &cold.stage_latent_mods_fp16,
            &cold.stage_latent_ctrl_ultra_fp16,
            is_ultra,
            (&mut cold.quad_adaa_scale, &mut cold.quad_lru_bias),
        );
        cold.op_n_pre_hoisted = hoisted;
    }

    slab.lite.op_n_pre.apply_hoisted(hoisted);

    let mut inc_core = [0_u32; 4];
    for (l, dst) in inc_core.iter_mut().enumerate() {
        let inc_vm = eval_phasor_inc_vm_q32(base_f[l], t[29], fs_vm, inv_fs_vm);
        *dst = eval_phasor_inc_core_multirate_q32(inc_vm, m_core) as u32;
    }
    slab.lite.phasors.inc = inc_core;

    let slots = TransportModSlots::from_targets_32(&t);
    let active_q = slab.cold().active_transport_quads.min(TRANSPORT_MAX_QUADS);
    for q in 0..active_q {
        let mut quad = *slab.transport_quad_hot(q);
        {
            let cold = slab.cold_mut();
            let meta = &cold.transport_quad_meta[q];
            let eff = &mut cold.transport_quad_eff[q];
            latch_quad_nonslew_targets(
                &mut quad,
                meta,
                &slots,
                cold.transport_ext.get_mut(q),
                cold.transport_schroeder.get_mut(q),
                eff,
            );
        }
        *slab.transport_quad_hot_mut(q) = quad;
    }
}

/// Hoists latched `ControlBus` `(current_stage_params, eff_z_stage)` targets into hot
/// `Op-N[Pre]`, Crossbar gains/phasor increments, `Transport` dry mix and non-slew targets, and
/// per-lane `CoreQuad` modulation arrays (`0` slew step).
#[inline(never)]
pub fn hoist_core_quads_and_readouts<B: GraphBackend>(slab: &mut SonaHotSlab<'_, B>) {
    hoist_core_quads_and_readouts_inline(slab);
}

#[inline(always)]
fn step_control_tick_3khz_inner<B: GraphBackend>(
    slab: &mut SonaHotSlab<'_, B>,
    immediate_snap: bool,
) {
    let t = slab.cold().latched_stage_params;
    let z = slab.cold().latched_z_stage;
    let dims = slab.lite.tier.stage_latent_dim();
    let [_, _, k_oct, k_damp, tank_psi_delta] =
        eval_global_stage_mods(&slab.cold().stage_latent_mods_fp16, &t, &z, dims);

    let active_banks = slab.cold().active_kautz_blocks;
    let mut banks_tmp = [Default::default(); MAX_KAUTZ_BANKS];
    for b in 0..active_banks {
        banks_tmp[b] = *slab.kautz_bank_mut(b);
    }
    let mut slew = slab.cold().kautz_slew;
    retune_kautz_banks_3khz::<B>(
        &mut banks_tmp[..active_banks],
        active_banks,
        &mut slew,
        k_oct,
        k_damp,
        immediate_snap,
    );
    slab.cold_mut().kautz_slew = slew;
    for b in 0..active_banks {
        *slab.kautz_bank_mut(b) = banks_tmp[b];
    }

    let active_q = slab.cold().active_transport_quads.min(TRANSPORT_MAX_QUADS);
    for q in 0..active_q {
        let mut quad = *slab.transport_quad_hot(q);
        {
            let cold = slab.cold_mut();
            let meta = &cold.transport_quad_meta[q];
            let eff = &mut cold.transport_quad_eff[q];
            let target_psi = eval_transport_target_psi(meta.base_psi, tank_psi_delta);
            if immediate_snap {
                quad.current_psi = target_psi;
            } else {
                let step_psi =
                    (target_psi - quad.current_psi).clamp(-PSI_SLEW_MAX_RAD, PSI_SLEW_MAX_RAD);
                quad.current_psi = (quad.current_psi + step_psi).clamp(-PI, PI);
            }
            (quad.cos_psi, quad.sin_psi) = algebraic_unit_rotor_f32(quad.current_psi);
            (eff.cos_psi, eff.sin_psi) = (quad.cos_psi, quad.sin_psi);
        }
        *slab.transport_quad_hot_mut(q) = quad;
    }

    hoist_core_quads_and_readouts_inline(slab);
}

/// Always-inlined `3 kHz` boundary control tick for `#[target_feature]` block dispatch.
#[inline(always)]
pub(crate) fn step_control_tick_3khz_inline<B: GraphBackend>(slab: &mut SonaHotSlab<'_, B>) {
    step_control_tick_3khz_inner(slab, false);
}

/// Executes one `3 kHz` boundary control tick (`Kautz` slew, `TransportQuad` Givens `psi` slew,
/// plus hot `CoreQuad`/readout hoisting).
#[inline(never)]
pub fn step_control_tick_3khz<B: GraphBackend>(slab: &mut SonaHotSlab<'_, B>) {
    step_control_tick_3khz_inner(slab, false);
}

/// Immediately latches current `ControlBus` targets and `eff_z_stage` and snaps `Kautz` and
/// `Transport` slews at init, tier switch, rate re-timing, or deterministic reset.
#[inline(never)]
pub fn latch_immediate_at_init_or_reset<B: GraphBackend>(slab: &mut SonaHotSlab<'_, B>) {
    if let Some(bus) = slab.cold_mut().ctrl_bus.as_mut() {
        bus.eval_static_and_snap_if_dirty();
        let stage = bus.current_stage_params;
        let zs = bus.eff_z_stage();
        let cold = slab.cold_mut();
        cold.latched_stage_params = stage;
        cold.latched_z_stage = zs;
    }
    step_control_tick_3khz_inner(slab, true);
    let cold = slab.cold_mut();
    let snap = cold.master_ctrl_snapshot();
    cold.master_os.ctrl_delay_ring.seed(snap);
    cold.master_os.active_master_core_ctrl = snap;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bin::{
        ARENA0_LITE_BYTES, DNBN_FLAG_STEREO_OUT, DnamBinBuildInput, DnamBinView,
        SONA_META_GRA_CONTROL_BUS, SONA_META_OS_EXT_PRESENT, encode_dnam_bin_into,
    };
    use crate::control::{ControlBusBacking, ControlClampDescriptor, Cp1ContributorSlot};
    use crate::graph::oversample::{MasterCtrlSnapshotCold, RenderMode};
    use crate::graph::slab::hydrate::{
        hydrate_slab_from_dnam_bin, reset_to_dc_equilibrium, set_render_mode,
    };
    use crate::graph::slab::layout::{MAX_CORE_QUADS, SonaColdTail, SonaSlabBacking};
    use crate::graph::slab::microchunk::{process_block_sona, step_frame_sona};
    use crate::math::DefaultMathBackend;
    use zerocopy::IntoBytes;
    use zerocopy::byteorder::little_endian::{F32, U16, U32};

    fn build_master_seed_test_bin() -> ([u8; 128 + ARENA0_LITE_BYTES + 64], usize) {
        let mut a0 = DnamBinBuildInput::default_arena0_lite();
        let meta = SONA_META_OS_EXT_PRESENT | SONA_META_GRA_CONTROL_BUS;
        a0.crossbar.sona_meta_flags = U32::new(meta);
        a0.crossbar.cell_desc.cell_flags = U32::new(meta);
        a0.crossbar.cell_desc.tier_flags = U16::new(0x0001);
        a0.crossbar.cell_desc.oversample_factor = U16::new(2);
        a0.crossbar.crossbar_op_gains_fp16 = [U16::new(f32_to_fp16(0.5)), U16::new(0)];
        a0.crossbar.phasor_freq_hz = [110.0, 220.0, 330.0, 440.0].map(F32::new);
        a0.crossbar.carrier_basis_w_fp16 = [[U16::new(f32_to_fp16(0.25)); 4]; 3];
        a0.crossbar.op_n_pre_base.gamma_bc = F32::new(0.2);

        a0.core_quads[0].svf_g = [0.15, 0.25, 0.45, 0.70].map(F32::new);
        a0.core_quads[0].svf_k_fp16 = [U16::new(f32_to_fp16(0.7)); 4];
        a0.core_quads[0].adaa_drive_fp16 = [U16::new(f32_to_fp16(1.2)); 4];
        a0.core_quads[0].lru_bias_drive_fp16 = [U16::new(f32_to_fp16(0.25)); 4];
        a0.core_quads[0].lru_alpha_rel = [F32::new(0.90); 4];
        a0.core_quads[0].lru_alpha_atk = [F32::new(0.25); 4];
        a0.core_quads[0].c_lp_fp16 = [U16::new(f32_to_fp16(0.4)); 4];
        a0.core_quads[0].c_u_fp16 = [U16::new(f32_to_fp16(0.6)); 4];
        a0.core_quads[0].in_mid_weight_fp16 = [U16::new(f32_to_fp16(0.5)); 4];

        a0.linear_readout_lite.direct_s2_u_gain = [F32::new(0.6), F32::new(0.6)];
        for l in 0..16 {
            a0.linear_readout_lite.w_ro_fp16[l * 12] = U16::new(f32_to_fp16(0.15));
            a0.linear_readout_lite.w_ro_fp16[l * 12 + 1] = U16::new(f32_to_fp16(0.15));
            a0.linear_readout_lite.w_ro_fp16[l * 12 + 6] = U16::new(f32_to_fp16(0.20));
            a0.linear_readout_lite.w_ro_fp16[l * 12 + 7] = U16::new(f32_to_fp16(0.20));
        }
        a0.linear_readout_lite.stage_latent_mods_fp16[80..96].fill(U16::new(f32_to_fp16(0.5)));
        a0.linear_readout_lite.w_ro_fp16[192..208].fill(U16::new(f32_to_fp16(0.5)));

        // Control 0 defaults to 0.85 -> non-zero norm & stage_targets[2, 6, 15] at t = 0.
        a0.control_bus_lite.cp1_dir.cp1_dir_count = U16::new(1);
        a0.control_bus_lite.cp1_dir.entries[0] = [0, 0x01, 0xFF];
        a0.control_bus_lite.gra_lite[..4].copy_from_slice(F32::new(1.0).as_bytes());
        a0.control_bus_lite.gra_lite[32..36].copy_from_slice(F32::new(1.0).as_bytes());
        a0.control_bus_lite.gra_lite[36..40].copy_from_slice(F32::new(1.0).as_bytes());
        a0.control_bus_lite.w_ctrl[2][4] = F32::new(0.6);
        a0.control_bus_lite.w_ctrl[2][8] = F32::new(0.25);
        a0.control_bus_lite.w_ctrl[6][4] = F32::new(-0.5);
        a0.control_bus_lite.w_ctrl[6][8] = F32::new(-0.2);
        a0.control_bus_lite.w_ctrl[15][4] = F32::new(0.8);
        a0.control_bus_lite.w_ctrl[15][8] = F32::new(0.35);

        let desc = ControlClampDescriptor::range(0.0, 1.0, 0.85, 1, [0.0; 3]).to_wire();
        let input = DnamBinBuildInput {
            reference_sample_rate_hz: 48_000,
            header_flags: DNBN_FLAG_STEREO_OUT,
            ext_ring_pow2_frames: 0,
            ext_ring_lines: 0,
            ext_ring_format: 0,
            declared_max_delay_48k: 128,
            dc_blocker_pole_5hz: crate::math::TANK_DC_BLOCKER_48K_F32,
            tail_time_ms: 0.0,
            ro_pole_48k: 0.99,
            switch_xfade_samples: 0,
            content_hash_prefix: [0; 8],
            controls: &[desc],
            cp1_overflow: &[],
            tsyn: &[],
            cext: None,
            arena0: &a0,
            arena1: None,
            arena2: None,
            arena3: None,
        };
        let mut buf = [0_u8; 128 + ARENA0_LITE_BYTES + 64];
        let n = encode_dnam_bin_into(&input, &mut buf).unwrap();
        (buf, n)
    }

    #[test]
    fn test_compute_global_mods_clamps_non_finite_and_phasor_q32() {
        let mut mods = [0_u16; 96];
        mods[8] = f32_to_fp16(4.0);
        let mut t = [0.0_f32; 32];
        t[0] = 2.0;
        t[14] = f32::NAN;
        let z = [1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let [g_oct, _, _, _, psi] = eval_global_stage_mods(&mods, &t, &z, 4);
        assert_eq!(g_oct, 3.0);
        assert_eq!(psi, 0.0);
        assert_eq!(MAX_CORE_QUADS, 10);

        let inv_48k = crate::math::rcp_nr6_f64(48_000.0);
        let inc_1x = eval_phasor_inc_vm_q32(-12_000.0, 0.0, 48_000.0, inv_48k);
        assert_eq!(inc_1x, -1_073_741_824);
        assert_eq!(eval_phasor_inc_core_multirate_q32(inc_1x, 2), -536_870_912);
        assert_eq!(eval_phasor_inc_core_multirate_q32(inc_1x, 4), -268_435_456);
        assert_eq!(eval_transport_dry_mix_stage(0.75, 0.5), 1.25);
    }

    struct StackSlabBufs<'a> {
        cold: [SonaColdTail<'a, DefaultMathBackend>; 1],
        descs: [ControlClampDescriptor; 1],
        raw: [f32; 1],
        cl: [f32; 1],
        norm: [f32; 1],
        sw: [u8; 1],
        con: [Cp1ContributorSlot; 8],
    }

    impl<'a> StackSlabBufs<'a> {
        fn new() -> Self {
            Self {
                cold: [SonaColdTail::empty()],
                descs: [ControlClampDescriptor::empty(); 1],
                raw: [0.0; 1],
                cl: [0.0; 1],
                norm: [0.0; 1],
                sw: [0; 1],
                con: [Cp1ContributorSlot::zero(); 8],
            }
        }
    }

    fn hydrate_stack_slab<'a>(
        view: &DnamBinView<'_>,
        offline: bool,
        bufs: &'a mut StackSlabBufs<'a>,
    ) -> SonaHotSlab<'a, DefaultMathBackend> {
        let ctrl = ControlBusBacking::from_slices(
            &mut bufs.descs,
            &mut bufs.raw,
            &mut bufs.cl,
            &mut bufs.norm,
            &mut bufs.sw,
            &mut bufs.con,
            &[],
        )
        .unwrap();
        hydrate_slab_from_dnam_bin::<DefaultMathBackend>(
            view,
            Tier::Lite,
            48_000.0,
            offline,
            SonaSlabBacking {
                control_bus: ctrl,
                convolver_fir_ext: &mut [],
                transport_ext_fp16: [&mut [], &mut [], &mut [], &mut []],
                transport_ext_f32: [&mut [], &mut [], &mut [], &mut []],
                transport_schroeder: [&mut [], &mut [], &mut [], &mut []],
                cold_tail: &mut bufs.cold,
            },
        )
        .unwrap()
    }

    #[test]
    fn test_master_4x_seeds_ctrl_delay_ring_and_routes_slot15_via_delayed_snapshot() {
        let (buf, n) = build_master_seed_test_bin();
        let view = DnamBinView::parse(&buf[..n]).unwrap();

        let mut bufs_a = StackSlabBufs::new();
        let mut slab_a = hydrate_stack_slab(&view, true, &mut bufs_a);

        let initial_snap = slab_a.cold().master_ctrl_snapshot();
        assert_ne!(initial_snap, MasterCtrlSnapshotCold::default());
        assert_ne!(initial_snap.latched_stage_params[15], 0.0);
        assert_ne!(initial_snap.quad_adaa_scale[0], 1.0);
        assert_ne!(initial_snap.quad_lru_bias[0], 0.0);
        assert_eq!(
            slab_a.cold().master_os.active_master_core_ctrl,
            initial_snap
        );
        assert_eq!(
            slab_a.cold().master_os.ctrl_delay_ring.slots,
            [initial_snap; 4]
        );

        // Hydrate an identical slab for quad-major block processing and verify 0..16 frames stay
        // seeded (never MasterCtrlSnapshotCold::default()) and match step_frame_sona 0-ULP.
        let mut bufs_b = StackSlabBufs::new();
        let mut slab_b = hydrate_stack_slab(&view, true, &mut bufs_b);

        let in_l = [0.25_f32; 16];
        let in_r = [-0.20_f32; 16];
        let (mut out_blk_l, mut out_blk_r) = ([0.0_f32; 16], [0.0_f32; 16]);
        process_block_sona(&mut slab_b, &in_l, &in_r, &mut out_blk_l, &mut out_blk_r).unwrap();
        assert_eq!(
            slab_b.cold().master_os.active_master_core_ctrl,
            initial_snap
        );

        for f in 0..16 {
            let (yl, yr) = step_frame_sona(&mut slab_a, in_l[f], in_r[f]).unwrap();
            assert_eq!(
                slab_a.cold().master_os.active_master_core_ctrl,
                initial_snap
            );
            assert_eq!(yl.to_bits(), out_blk_l[f].to_bits());
            assert_eq!(yr.to_bits(), out_blk_r[f].to_bits());
        }

        // Verify latched_stage_params[15] uses delayed act.latched_stage_params[15] for frames
        // 0..15 and only takes effect at frame 16 (delta_D_up = 16).
        reset_to_dc_equilibrium(&mut slab_a);
        reset_to_dc_equilibrium(&mut slab_b);
        slab_a.cold_mut().ctrl_bus = None;
        slab_b.cold_mut().ctrl_bus = None;
        slab_b.cold_mut().latched_stage_params[15] = -0.95;

        let mut step_out_l = [0.0_f32; 16];
        let mut step_out_r = [0.0_f32; 16];
        for f in 0..16 {
            let (al, ar) = step_frame_sona(&mut slab_a, in_l[f], in_r[f]).unwrap();
            let (bl, br) = step_frame_sona(&mut slab_b, in_l[f], in_r[f]).unwrap();
            step_out_l[f] = bl;
            step_out_r[f] = br;
            assert_eq!(
                slab_b
                    .cold()
                    .master_os
                    .active_master_core_ctrl
                    .latched_stage_params[15],
                initial_snap.latched_stage_params[15]
            );
            assert_eq!(
                al.to_bits(),
                bl.to_bits(),
                "frame {f} L diverged prematurely"
            );
            assert_eq!(
                ar.to_bits(),
                br.to_bits(),
                "frame {f} R diverged prematurely"
            );
        }
        // Frame 16 pops the modified slot[15] = -0.95 pushed at chunk 0.
        let (al16, _) = step_frame_sona(&mut slab_a, in_l[0], in_r[0]).unwrap();
        let (bl16, _) = step_frame_sona(&mut slab_b, in_l[0], in_r[0]).unwrap();
        assert_eq!(
            slab_b
                .cold()
                .master_os
                .active_master_core_ctrl
                .latched_stage_params[15],
            -0.95
        );
        assert_ne!(al16.to_bits(), bl16.to_bits());

        // Also check quad_major block path matches step_frame_sona 0-ULP over frames 0..16 with
        // modified cold.latched_stage_params[15].
        slab_b.cold_mut().latched_stage_params[15] = initial_snap.latched_stage_params[15];
        reset_to_dc_equilibrium(&mut slab_b);
        slab_b.cold_mut().latched_stage_params[15] = -0.95;
        process_block_sona(&mut slab_b, &in_l, &in_r, &mut out_blk_l, &mut out_blk_r).unwrap();
        for f in 0..16 {
            assert_eq!(out_blk_l[f].to_bits(), step_out_l[f].to_bits());
            assert_eq!(out_blk_r[f].to_bits(), step_out_r[f].to_bits());
        }

        // Re-switching Live -> Master via set_render_mode re-seeds cleanly with zero allocs.
        let mut bufs_c = StackSlabBufs::new();
        let mut slab_c = hydrate_stack_slab(&view, false, &mut bufs_c);
        for f in 0..12 {
            let _ = step_frame_sona(&mut slab_c, in_l[f], in_r[f]).unwrap();
        }
        set_render_mode(&mut slab_c, RenderMode::Master).unwrap();
        assert_eq!(
            slab_c.cold().master_os.active_master_core_ctrl,
            initial_snap
        );
        assert_eq!(
            slab_c.cold().master_os.ctrl_delay_ring.slots,
            [initial_snap; 4]
        );
        for f in 0..16 {
            let _ = step_frame_sona(&mut slab_c, in_l[f], in_r[f]).unwrap();
            assert_eq!(
                slab_c.cold().master_os.active_master_core_ctrl,
                initial_snap
            );
        }
    }
}
