//! Matryoshka binary arena ownership (`Arena0..3` / Readouts `4`) and readout-coupling masks
//! for [`ParamKind`] (`ref[train.pipeline.backbone-trajectory-opt]`).

use super::ParamKind;

#[inline]
const fn core_quad_arena(q: usize) -> u8 {
    match q {
        0..=3 => 0,
        4..=5 => 1,
        6..=7 => 2,
        _ => 3,
    }
}

#[inline]
const fn tile_arena(t: usize) -> u8 {
    match t {
        0..=2 => 0,
        3..=12 => 1,
        13..=22 => 2,
        _ => 3,
    }
}

#[inline]
const fn kautz_arena(k: usize) -> u8 {
    match k {
        0..=7 => 0,
        8..=11 => 1,
        12..=23 => 2,
        _ => 3,
    }
}

const _: () = assert!(
    core_quad_arena(0) == 0
        && core_quad_arena(3) == 0
        && core_quad_arena(4) == 1
        && core_quad_arena(5) == 1
        && core_quad_arena(6) == 2
        && core_quad_arena(7) == 2
        && core_quad_arena(8) == 3
        && core_quad_arena(9) == 3
);
const _: () = assert!(
    tile_arena(0) == 0
        && tile_arena(2) == 0
        && tile_arena(3) == 1
        && tile_arena(12) == 1
        && tile_arena(13) == 2
        && tile_arena(22) == 2
        && tile_arena(23) == 3
        && tile_arena(34) == 3
);
const _: () = assert!(
    kautz_arena(0) == 0
        && kautz_arena(7) == 0
        && kautz_arena(8) == 1
        && kautz_arena(11) == 1
        && kautz_arena(12) == 2
        && kautz_arena(23) == 2
        && kautz_arena(24) == 3
        && kautz_arena(31) == 3
);

impl ParamKind {
    /// Returns the Matryoshka binary arena ID (`0..=3` for `Arena0..Arena3`, `4` for Readouts).
    #[inline]
    #[must_use]
    pub(in crate::solver::pass2_opt::theta) const fn arena_id(self) -> u8 {
        match self {
            Self::CoreSvfG(q, _)
            | Self::CoreSvfK(q, _)
            | Self::CoreSvfKNl(q, _)
            | Self::CoreToneLp(q, _)
            | Self::CoreToneBp(q, _)
            | Self::CoreToneHp(q, _)
            | Self::CoreToneAp(q, _)
            | Self::CoreToneDir(q, _)
            | Self::CoreLruRel(q, _)
            | Self::CoreLruAtk(q, _)
            | Self::CoreLruKap(q, _)
            | Self::CoreLruDuhem(q, _)
            | Self::CoreLruSignedMix(q, _)
            | Self::CoreLruStereoLink(q, _)
            | Self::CoreSagDepth(q, _)
            | Self::CoreAdaaDrive(q, _)
            | Self::CoreLruBias(q, _)
            | Self::CoreWavefolder(q, _)
            | Self::CoreGluMix(q, _)
            | Self::CoreG0(q, _)
            | Self::CoreGInf(q, _)
            | Self::CoreDPre(q, _)
            | Self::CoreSeWSq(q, _)
            | Self::CoreSeDBus(q, _)
            | Self::CoreLinMix(q, _)
            | Self::CoreOutGain(q, _)
            | Self::CoreInMid(q, _)
            | Self::CoreInSide(q, _)
            | Self::OpPToCore(q, _)
            | Self::PhasorToCore(q, _) => core_quad_arena(q),
            Self::TileOdd(t, _, _) | Self::TileEven(t, _, _) | Self::TileLru(t, _, _) => {
                tile_arena(t)
            }
            Self::SenseSvfG(_)
            | Self::SenseLruRel(_)
            | Self::SenseLruAtk(_)
            | Self::SenseSvfK(_)
            | Self::SenseToneLp(_)
            | Self::SenseToneBp(_)
            | Self::SenseToneHp(_)
            | Self::SenseLruKappa(_)
            | Self::SenseVcaW(_)
            | Self::SenseInL(_)
            | Self::SenseInR(_)
            | Self::PhasorFreqHz(_)
            | Self::CrossbarOpGain(_)
            | Self::CarrierBasisW(_, _)
            | Self::OpNPreGammaBc
            | Self::OpNPreZohHoldDepth
            | Self::OpNPreBetaQ
            | Self::RoPole48k
            | Self::TankDryMix => 0,
            Self::PhasorStereoPhase(_) => 1,
            Self::OptToLr(i, _) => {
                let a = i / 4;
                if a > 3 { 3 } else { a as u8 }
            }
            Self::KautzRLow(k) | Self::KautzTheta(k) | Self::KautzBGamma(k) => kautz_arena(k),
            Self::OpTDelay(q, _)
            | Self::OpTRampSpan(q, _)
            | Self::OpTGLoop(q)
            | Self::OpTRDc(q)
            | Self::OpTDampPole(q)
            | Self::OpTLfoRate(q)
            | Self::OpTRampRate(q)
            | Self::OpTLfoDepth(q, _)
            | Self::OpTInputGain(q, _)
            | Self::OpTAllpassG(q, _)
            | Self::OpTFbPsi(q)
            | Self::OpTSigmaKnee(q)
            | Self::OpTFreezeMix(q)
            | Self::OpTWindowDepth(q)
            | Self::OpTScatPsi(q)
            | Self::OpTWetTapL(q, _)
            | Self::OpTWetTapR(q, _) => {
                if q > 3 {
                    3
                } else {
                    q as u8
                }
            }
            Self::WReadout(_, _)
            | Self::DirectS2U(_)
            | Self::DirectS2Hp(_)
            | Self::PreConvU(_)
            | Self::PreConvHp(_)
            | Self::KautzCReL(_)
            | Self::KautzCImL(_)
            | Self::KautzCReR(_)
            | Self::KautzCImR(_) => 4,
        }
    }

    /// Returns `true` when this coordinate is directly coupled to linear VarPro readouts so
    /// periodic VarPro refreshes reset its AdamW first/second moments cleanly.
    #[inline]
    #[must_use]
    pub(in crate::solver::pass2_opt::theta) const fn is_readout_coupled(self) -> bool {
        matches!(
            self,
            Self::RoPole48k
                | Self::KautzRLow(_)
                | Self::KautzTheta(_)
                | Self::KautzBGamma(_)
                | Self::WReadout(_, _)
                | Self::DirectS2U(_)
                | Self::DirectS2Hp(_)
                | Self::PreConvU(_)
                | Self::PreConvHp(_)
                | Self::KautzCReL(_)
                | Self::KautzCImL(_)
                | Self::KautzCReR(_)
                | Self::KautzCImR(_)
        )
    }
}
