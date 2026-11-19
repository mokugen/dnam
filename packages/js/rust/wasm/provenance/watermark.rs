//! WASM v2 isometric gauge / Tardos buyer watermarking and structural fingerprinting exports
//! (`ref[sdk.ts.provenance.v2-security]`, `ref[sdk.ts.runtime.zero-unsafe]`).

use crate::core::provenance::{
    apply_isometric_gauge_watermark_core, apply_tardos_watermark_core,
    derive_watermark_seed_hex_core, extract_stage3_fingerprint_core, extract_structural_hash_core,
    generate_tardos_code_core, score_tardos_watermark_core, structural_cosine_similarity_core,
};
use crate::error::validation_err;
use crate::wasm::abi::{
    read_bytes, read_f32_vec, read_f64_vec, read_opt_bytes, read_str, set_error, set_result,
    with_f32_mut, with_f64_mut, write_f64,
};

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_apply_isometric_gauge_watermark(
    arena_ptr: usize,
    arena_len: usize,
    kappa_ptr: usize,
    kappa_len: usize,
    q_ptr: usize,
    q_len: usize,
) -> i32 {
    let arena = read_bytes(arena_ptr, arena_len);
    let kappa = read_f32_vec(kappa_ptr, kappa_len);
    let q = read_f32_vec(q_ptr, q_len);
    match apply_isometric_gauge_watermark_core(&arena, &kappa, &q) {
        Ok(bytes) => set_result(bytes),
        Err(msg) => set_error(msg),
    }
}

#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_generate_tardos_code(
    sec_ptr: usize,
    sec_len: usize,
    model_ptr: usize,
    model_len: usize,
    lic_ptr: usize,
    lic_len: usize,
    out_ptr: usize,
    out_len: usize,
) -> i32 {
    let sec = read_bytes(sec_ptr, sec_len);
    let model = match read_str(model_ptr, model_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let lic = match read_str(lic_ptr, lic_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    match generate_tardos_code_core(&sec, &model, &lic) {
        Ok(code) => {
            let Some(ok) = with_f32_mut(out_ptr, out_len, |out| {
                if out.len() < code.len() {
                    return false;
                }
                out[..code.len()].copy_from_slice(&code);
                true
            }) else {
                return set_error("[DNAM-VALIDATION] invalid out_ptr".to_string());
            };
            if ok {
                0
            } else {
                set_error(validation_err("out buffer too short for Tardos code"))
            }
        }
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_apply_tardos_watermark(
    arena_ptr: usize,
    arena_len: usize,
    code_ptr: usize,
    code_len: usize,
    epsilon: f32,
) -> i32 {
    let arena = read_bytes(arena_ptr, arena_len);
    let code = read_f32_vec(code_ptr, code_len);
    match apply_tardos_watermark_core(&arena, &code, epsilon) {
        Ok(bytes) => set_result(bytes),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_score_tardos_watermark(
    master_ptr: usize,
    master_len: usize,
    suspect_ptr: usize,
    suspect_len: usize,
    code_ptr: usize,
    code_len: usize,
    out_score_ptr: usize,
) -> i32 {
    if out_score_ptr == 0 {
        return set_error("[DNAM-VALIDATION] null out_score_ptr".to_string());
    }
    let master = read_bytes(master_ptr, master_len);
    let suspect = read_bytes(suspect_ptr, suspect_len);
    let code = read_f32_vec(code_ptr, code_len);
    match score_tardos_watermark_core(&master, &suspect, &code) {
        Ok(score) => {
            if !write_f64(out_score_ptr, score) {
                return set_error("[DNAM-VALIDATION] invalid out_score_ptr".to_string());
            }
            0
        }
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_extract_structural_hash(
    arena_ptr: usize,
    arena_len: usize,
    out_ptr: usize,
    out_cap: usize,
) -> i32 {
    let arena = read_bytes(arena_ptr, arena_len);
    match extract_structural_hash_core(&arena) {
        Ok(vec) => {
            let Some(ok) = with_f64_mut(out_ptr, out_cap, |out| {
                if out.len() < vec.len() {
                    return false;
                }
                out[..vec.len()].copy_from_slice(&vec);
                true
            }) else {
                return set_error("[DNAM-VALIDATION] invalid out_ptr".to_string());
            };
            if ok {
                vec.len() as i32
            } else {
                set_error(validation_err("out buffer too short for structural hash"))
            }
        }
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_structural_cosine_similarity(
    a_ptr: usize,
    a_len: usize,
    b_ptr: usize,
    b_len: usize,
    out_sim_ptr: usize,
) -> i32 {
    if out_sim_ptr == 0 {
        return set_error("[DNAM-VALIDATION] null out_sim_ptr".to_string());
    }
    let a = read_f64_vec(a_ptr, a_len);
    let b = read_f64_vec(b_ptr, b_len);
    match structural_cosine_similarity_core(&a, &b) {
        Ok(sim) => {
            if !write_f64(out_sim_ptr, sim) {
                return set_error("[DNAM-VALIDATION] invalid out_sim_ptr".to_string());
            }
            0
        }
        Err(msg) => set_error(msg),
    }
}

#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_derive_watermark_seed_hex(
    key_ptr: usize,
    key_len: usize,
    mkt_pk_ptr: usize,
    mkt_pk_len: usize,
    wm_kid_ptr: usize,
    wm_kid_len: usize,
    man_sha_ptr: usize,
    man_sha_len: usize,
    model_id_ptr: usize,
    model_id_len: usize,
    prof_sha_ptr: usize,
    prof_sha_len: usize,
    owning_tier: u8,
    lic_ptr: usize,
    lic_len: usize,
    buyer_ptr: usize,
    buyer_len: usize,
) -> i32 {
    let key = read_bytes(key_ptr, key_len);
    let mkt_pk = read_bytes(mkt_pk_ptr, mkt_pk_len);
    let wm_kid = read_bytes(wm_kid_ptr, wm_kid_len);
    let man_sha = read_bytes(man_sha_ptr, man_sha_len);
    let model_id = read_bytes(model_id_ptr, model_id_len);
    let prof_sha = read_bytes(prof_sha_ptr, prof_sha_len);
    let lic = read_opt_bytes(lic_ptr, lic_len);
    let buyer = read_opt_bytes(buyer_ptr, buyer_len);
    match derive_watermark_seed_hex_core(
        &key,
        &mkt_pk,
        &wm_kid,
        &man_sha,
        &model_id,
        &prof_sha,
        owning_tier,
        lic.as_deref(),
        buyer.as_deref(),
    ) {
        Ok(hex) => set_result(hex.into_bytes()),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_extract_stage3_fingerprint(
    fir_ptr: usize,
    fir_len: usize,
    kautz_ptr: usize,
    kautz_len: usize,
    added_modes: usize,
    out_ptr: usize,
    out_cap: usize,
) -> i32 {
    let fir = read_bytes(fir_ptr, fir_len);
    let kautz = read_bytes(kautz_ptr, kautz_len);
    match extract_stage3_fingerprint_core(&fir, &kautz, added_modes) {
        Ok(vec) => {
            let Some(ok) = with_f64_mut(out_ptr, out_cap, |out| {
                if out.len() < vec.len() {
                    return false;
                }
                out[..vec.len()].copy_from_slice(&vec);
                true
            }) else {
                return set_error("[DNAM-VALIDATION] invalid out_ptr".to_string());
            };
            if ok {
                vec.len() as i32
            } else {
                set_error(validation_err(
                    "out buffer too short for Stage III fingerprint",
                ))
            }
        }
        Err(msg) => set_error(msg),
    }
}
