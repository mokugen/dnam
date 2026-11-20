//! WASM v2 container sealing/unsealing, `DAU1`/`DME1` envelopes, HPKE directories, preview
//! containment, and royalty split exports (`ref[sdk.ts.provenance.security-and-drm]`,
//! `ref[sdk.ts.runtime.zero-unsafe]`).

use crate::core::provenance::{
    build_dme1_header_core, chacha20_xor_arena_core, compute_preview_window_core,
    decode_dau1_envelope_core, encode_dau1_envelope_core, encode_hpke_slots_core,
    encode_recipient_key_ids_core, encode_record_directory_core, parse_dme1_plaintext_core,
    parse_hpke_slots_core, parse_recipient_key_ids_core, parse_record_directory_core,
    seal_container_core, unseal_container_core, validate_preview_switch_options_core,
    validate_royalty_splits_core, wrap_device_cek_hex_core,
};
use crate::error::validation_err;
use crate::wasm::abi::{
    read_bytes, read_opt_bytes, read_opt_str, read_str, set_error, set_result, write_f64_slice,
};

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_chacha20_xor_arena(
    arena_ptr: usize,
    arena_len: usize,
    key_ptr: usize,
    key_len: usize,
    nonce_ptr: usize,
    nonce_len: usize,
    start_block_counter: u32,
) -> i32 {
    let arena = read_bytes(arena_ptr, arena_len);
    let key = read_bytes(key_ptr, key_len);
    let nonce = read_bytes(nonce_ptr, nonce_len);
    match chacha20_xor_arena_core(&arena, &key, &nonce, start_block_counter) {
        Ok(bytes) => set_result(bytes),
        Err(msg) => set_error(msg),
    }
}

#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_wrap_device_cek_hex(
    eph_ptr: usize,
    eph_len: usize,
    dev_pk_ptr: usize,
    dev_pk_len: usize,
    dev_id_ptr: usize,
    dev_id_len: usize,
    cek_ptr: usize,
    cek_len: usize,
) -> i32 {
    let eph = read_bytes(eph_ptr, eph_len);
    let dev_pk = read_bytes(dev_pk_ptr, dev_pk_len);
    let dev_id = match read_str(dev_id_ptr, dev_id_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let cek = read_bytes(cek_ptr, cek_len);
    match wrap_device_cek_hex_core(&eph, &dev_pk, &dev_id, &cek) {
        Ok(hex) => set_result(hex.into_bytes()),
        Err(msg) => set_error(msg),
    }
}

#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_seal_container(
    inner_ptr: usize,
    inner_len: usize,
    certs_ptr: usize,
    certs_len: usize,
    oem_pks_ptr: usize,
    oem_pks_len: usize,
    mkt_seed_ptr: usize,
    mkt_seed_len: usize,
    eph_sec_ptr: usize,
    eph_sec_len: usize,
    cek_ptr: usize,
    cek_len: usize,
    nonce_ptr: usize,
    nonce_len: usize,
) -> i32 {
    let inner = read_bytes(inner_ptr, inner_len);
    let certs = match read_str(certs_ptr, certs_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let oem_pks = match read_str(oem_pks_ptr, oem_pks_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let mkt_seed = read_bytes(mkt_seed_ptr, mkt_seed_len);
    let eph_sec = read_bytes(eph_sec_ptr, eph_sec_len);
    let cek = read_bytes(cek_ptr, cek_len);
    let nonce = read_bytes(nonce_ptr, nonce_len);
    match seal_container_core(&inner, &certs, &oem_pks, &mkt_seed, &eph_sec, &cek, &nonce) {
        Ok(bytes) => set_result(bytes),
        Err(msg) => set_error(msg),
    }
}

#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_unseal_container(
    sealed_ptr: usize,
    sealed_len: usize,
    dev_id_ptr: usize,
    dev_id_len: usize,
    dev_sec_ptr: usize,
    dev_sec_len: usize,
    mkt_pk_ptr: usize,
    mkt_pk_len: usize,
) -> i32 {
    let sealed = read_bytes(sealed_ptr, sealed_len);
    let dev_id = match read_opt_str(dev_id_ptr, dev_id_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let dev_sec = read_opt_bytes(dev_sec_ptr, dev_sec_len);
    let mkt_pk = match read_opt_str(mkt_pk_ptr, mkt_pk_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    match unseal_container_core(
        &sealed,
        dev_id.as_deref(),
        dev_sec.as_deref(),
        mkt_pk.as_deref(),
    ) {
        Ok(bytes) => set_result(bytes),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_encode_dau1_envelope(spec_ptr: usize, spec_len: usize) -> i32 {
    let spec = match read_str(spec_ptr, spec_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    match encode_dau1_envelope_core(&spec) {
        Ok(bytes) => set_result(bytes),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_decode_dau1_envelope(wire_ptr: usize, wire_len: usize) -> i32 {
    let wire = read_bytes(wire_ptr, wire_len);
    match decode_dau1_envelope_core(&wire) {
        Ok(json) => set_result(json.into_bytes()),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_build_dme1_header(
    profile: u8,
    recipient_count: u32,
    ciphertext_length: u64,
) -> i32 {
    match build_dme1_header_core(profile, recipient_count, ciphertext_length) {
        Ok(bytes) => set_result(bytes),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_parse_dme1_plaintext(
    pt_ptr: usize,
    pt_len: usize,
    digests_json_ptr: usize,
    digests_json_len: usize,
) -> i32 {
    let pt = read_bytes(pt_ptr, pt_len);
    let digests_str = match read_str(digests_json_ptr, digests_json_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let digests: Vec<String> = match serde_json::from_str(&digests_str) {
        Ok(v) => v,
        Err(e) => return set_error(validation_err(e.to_string())),
    };
    match parse_dme1_plaintext_core(&pt, &digests) {
        Ok(json) => set_result(json.into_bytes()),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_encode_recipient_key_ids(json_ptr: usize, json_len: usize) -> i32 {
    let json_str = match read_str(json_ptr, json_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let digests: Vec<String> = match serde_json::from_str(&json_str) {
        Ok(v) => v,
        Err(e) => return set_error(validation_err(e.to_string())),
    };
    match encode_recipient_key_ids_core(&digests) {
        Ok(bytes) => set_result(bytes),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_parse_recipient_key_ids(
    data_ptr: usize,
    data_len: usize,
    max_recipients: usize,
) -> i32 {
    let data = read_bytes(data_ptr, data_len);
    match parse_recipient_key_ids_core(&data, max_recipients) {
        Ok(list) => match serde_json::to_vec(&list) {
            Ok(vec) => set_result(vec),
            Err(e) => set_error(validation_err(e.to_string())),
        },
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_encode_hpke_slots(json_ptr: usize, json_len: usize) -> i32 {
    let json_str = match read_str(json_ptr, json_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let slots: Vec<(String, String)> = match serde_json::from_str(&json_str) {
        Ok(v) => v,
        Err(e) => return set_error(validation_err(e.to_string())),
    };
    match encode_hpke_slots_core(&slots) {
        Ok(bytes) => set_result(bytes),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_parse_hpke_slots(
    data_ptr: usize,
    data_len: usize,
    expected_count: usize,
    max_recipients: usize,
) -> i32 {
    let data = read_bytes(data_ptr, data_len);
    match parse_hpke_slots_core(&data, expected_count, max_recipients) {
        Ok(slots) => match serde_json::to_vec(&slots) {
            Ok(vec) => set_result(vec),
            Err(e) => set_error(validation_err(e.to_string())),
        },
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_encode_record_directory(
    maximum_tier: u8,
    json_ptr: usize,
    json_len: usize,
) -> i32 {
    let json_str = match read_str(json_ptr, json_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let records: Vec<(u64, u64)> = match serde_json::from_str(&json_str) {
        Ok(v) => v,
        Err(e) => return set_error(validation_err(e.to_string())),
    };
    match encode_record_directory_core(maximum_tier, &records) {
        Ok(bytes) => set_result(bytes),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_parse_record_directory(
    data_ptr: usize,
    data_len: usize,
    maximum_tier: u8,
) -> i32 {
    let data = read_bytes(data_ptr, data_len);
    match parse_record_directory_core(&data, maximum_tier) {
        Ok(records) => match serde_json::to_vec(&records) {
            Ok(vec) => set_result(vec),
            Err(e) => set_error(validation_err(e.to_string())),
        },
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_compute_preview_window(
    anchor: f64,
    delta: f64,
    safe_min: f64,
    safe_max: f64,
    out2_ptr: usize,
) -> i32 {
    if out2_ptr == 0 {
        return set_error("[DNAM-VALIDATION] null out2_ptr".to_string());
    }
    match compute_preview_window_core(anchor, delta, (safe_min, safe_max)) {
        Ok(res) => {
            if !write_f64_slice(out2_ptr, &res) {
                return set_error("[DNAM-VALIDATION] invalid out2_ptr".to_string());
            }
            0
        }
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_validate_preview_switch_options(
    json_ptr: usize,
    json_len: usize,
) -> i32 {
    let json_str = match read_str(json_ptr, json_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct Spec {
        preset: String,
        alternative: String,
        observed: Vec<String>,
        active: Vec<String>,
        safe: Vec<String>,
    }
    let spec: Spec = match serde_json::from_str(&json_str) {
        Ok(v) => v,
        Err(e) => return set_error(validation_err(e.to_string())),
    };
    match validate_preview_switch_options_core(
        &spec.preset,
        &spec.alternative,
        &spec.observed,
        &spec.active,
        &spec.safe,
    ) {
        Ok((p, a)) => match serde_json::to_vec(&[p, a]) {
            Ok(vec) => set_result(vec),
            Err(e) => set_error(validation_err(e.to_string())),
        },
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_validate_royalty_splits(
    tier_ptr: usize,
    tier_len: usize,
    splits_ptr: usize,
    splits_len: usize,
) -> i32 {
    let tier = match read_str(tier_ptr, tier_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let splits = match read_str(splits_ptr, splits_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    match validate_royalty_splits_core(&tier, &splits) {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}
