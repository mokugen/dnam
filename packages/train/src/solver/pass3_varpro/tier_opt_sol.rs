//! Per-tier stereo normal-equation assembly and bounded `uplo='U'` solve for Pass-3 `VarPro`.

use crate::error::TrainError;
use crate::linalg::varpro::assemble_ozaki_normal_equations;
use crate::runner::TrainRunner;

use super::super::model::SonaModelF64;
use super::eval_hp_post_for_ro;
use super::spectral_refine::solve_bounded_uplo_u;

pub(super) struct TierOptChannelSol {
    pub gl: Vec<f64>,
    pub rl: Vec<f64>,
    pub wl: Vec<f64>,
    pub rl_ridge: f64,
    pub gr: Vec<f64>,
    pub rr: Vec<f64>,
    pub wr: Vec<f64>,
    pub rr_ridge: f64,
}

pub(super) struct TierOptSolveInputs<'a> {
    pub fl: &'a mut [f32],
    pub fr: &'a mut [f32],
    pub tl: &'a [f32],
    pub tr: &'a [f32],
    pub raw_l: &'a [f64],
    pub raw_r: &'a [f64],
    pub raw_l_2x: Option<&'a [f64]>,
    pub raw_r_2x: Option<&'a [f64]>,
    pub dim_t: usize,
    pub active_opt: usize,
    pub w: usize,
    pub force_wide192: bool,
}

pub(super) fn solve_tier_opt_channel_sol(
    runner: &TrainRunner,
    model: &SonaModelF64,
    inp: TierOptSolveInputs<'_>,
) -> Result<TierOptChannelSol, TrainError> {
    let m = inp.tl.len();
    let hp_l = eval_hp_post_for_ro(
        model,
        inp.raw_l,
        inp.raw_l_2x,
        model.ro_pole_48k,
        inp.w,
        false,
        inp.active_opt,
    );
    let hp_r = eval_hp_post_for_ro(
        model,
        inp.raw_r,
        inp.raw_r_2x,
        model.ro_pole_48k,
        inp.w,
        true,
        inp.active_opt,
    );
    inp.fl[m..(2 * m)].copy_from_slice(&hp_l);
    inp.fr[m..(2 * m)].copy_from_slice(&hp_r);
    let (gl, rl_v) = assemble_ozaki_normal_equations(
        runner,
        inp.fl,
        inp.tl,
        m,
        inp.dim_t,
        1,
        inp.force_wide192,
    )?;
    let (gr, rr_v) = assemble_ozaki_normal_equations(
        runner,
        inp.fr,
        inp.tr,
        m,
        inp.dim_t,
        1,
        inp.force_wide192,
    )?;
    let rl: Vec<f64> = rl_v.into_iter().map(|v| v[0]).collect();
    let rr: Vec<f64> = rr_v.into_iter().map(|v| v[0]).collect();
    let (wl, rl_ridge) =
        solve_bounded_uplo_u(&gl, &rl, inp.dim_t, model.ro_pole_48k, model.sample_rate_hz)?;
    let (wr, rr_ridge) =
        solve_bounded_uplo_u(&gr, &rr, inp.dim_t, model.ro_pole_48k, model.sample_rate_hz)?;
    Ok(TierOptChannelSol {
        gl,
        rl,
        wl,
        rl_ridge,
        gr,
        rr,
        wr,
        rr_ridge,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve_tier_opt_channel_sol_reconstructs_weights() {
        let runner = TrainRunner::cpu();
        let model = SonaModelF64::new(48_000.0);
        let m = 32_usize;
        let dim_t = 4_usize;
        let mut fl = vec![0.0_f32; dim_t * m];
        let mut fr = vec![0.0_f32; dim_t * m];
        let mut tl = vec![0.0_f32; m];
        let mut tr = vec![0.0_f32; m];
        let mut raw_l = vec![0.0_f64; m];
        let mut raw_r = vec![0.0_f64; m];
        for i in 0..m {
            let x = ((i as f64) * 0.23).sin();
            let z = ((i as f64) * 0.41).cos();
            raw_l[i] = x;
            raw_r[i] = z;
            fl[i] = x as f32;
            fl[2 * m + i] = z as f32;
            fr[i] = z as f32;
            fr[2 * m + i] = x as f32;
            tl[i] = (0.6 * x) as f32;
            tr[i] = (0.5 * z) as f32;
        }
        let sol = solve_tier_opt_channel_sol(
            &runner,
            &model,
            TierOptSolveInputs {
                fl: &mut fl,
                fr: &mut fr,
                tl: &tl,
                tr: &tr,
                raw_l: &raw_l,
                raw_r: &raw_r,
                raw_l_2x: None,
                raw_r_2x: None,
                dim_t,
                active_opt: 1,
                w: 0,
                force_wide192: false,
            },
        )
        .unwrap();
        assert_eq!(sol.wl.len(), dim_t);
        assert_eq!(sol.wr.len(), dim_t);
        assert!(sol.wl.iter().chain(&sol.wr).all(|v| v.is_finite()));
    }
}
