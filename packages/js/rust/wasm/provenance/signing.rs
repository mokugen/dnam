//! WASM v2 provenance verification, hashing, Ed25519/X25519 signing, and OEM certificate
//! exports (`ref[sdk.ts.provenance.v2-security]`, `ref[sdk.ts.runtime.zero-unsafe]`).

use crate::core::provenance::{
    build_creator_signing_message_core, build_marketplace_signing_message_core,
    build_oem_cert_v2_signing_message_core, ed25519_pubkey_hex_core, ed25519_sign_hex_core,
    enforce_distribution_policy_core, evaluate_anticlone_policy_core, hmac_sha256_hex_core,
    immutable_header_sha256_core, sha256_hex_core, sign_creator_header_core,
    sign_marketplace_receipt_core, sign_oem_device_cert_core, verify_container_provenance_core,
    verify_oem_device_cert_core, x25519_pubkey_hex_core,
};
use crate::error::validation_err;
use crate::wasm::abi::{read_bytes, read_opt_bytes, read_opt_str, read_str, set_error, set_result};

#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_verify_container_provenance(
    hdr_ptr: usize,
    hdr_len: usize,
    arena_ptr: usize,
    arena_len: usize,
    has_sig: u8,
    dev_id_ptr: usize,
    dev_id_len: usize,
    dev_sec_ptr: usize,
    dev_sec_len: usize,
    mkt_pk_ptr: usize,
    mkt_pk_len: usize,
) -> i32 {
    let hdr = match read_str(hdr_ptr, hdr_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let arena = read_bytes(arena_ptr, arena_len);
    let dev_id = match read_opt_str(dev_id_ptr, dev_id_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let dev_sec = read_opt_bytes(dev_sec_ptr, dev_sec_len);
    let mkt_pk = match read_opt_str(mkt_pk_ptr, mkt_pk_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    match verify_container_provenance_core(
        &hdr,
        &arena,
        has_sig != 0,
        dev_id.as_deref(),
        dev_sec.as_deref(),
        mkt_pk.as_deref(),
    ) {
        Ok(info) => match serde_json::to_vec(&info) {
            Ok(vec) => set_result(vec),
            Err(e) => set_error(validation_err(e.to_string())),
        },
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_enforce_distribution_policy(
    hdr_ptr: usize,
    hdr_len: usize,
    prof_ptr: usize,
    prof_len: usize,
) -> i32 {
    let hdr = match read_str(hdr_ptr, hdr_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let prof = match read_str(prof_ptr, prof_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    match enforce_distribution_policy_core(&hdr, &prof) {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_evaluate_anticlone_policy(
    fingerprint_matched: u8,
    buyer_watermark_identified: u8,
    independent_device_source_evidence: u8,
) -> i32 {
    let s = evaluate_anticlone_policy_core(
        fingerprint_matched != 0,
        buyer_watermark_identified != 0,
        independent_device_source_evidence != 0,
    );
    set_result(s.as_bytes().to_vec())
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_immutable_header_sha256(hdr_ptr: usize, hdr_len: usize) -> i32 {
    let hdr = match read_str(hdr_ptr, hdr_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    match immutable_header_sha256_core(&hdr) {
        Ok(hex) => set_result(hex.into_bytes()),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_sha256_hex(data_ptr: usize, data_len: usize) -> i32 {
    let data = read_bytes(data_ptr, data_len);
    set_result(sha256_hex_core(&data).into_bytes())
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_hmac_sha256_hex(
    key_ptr: usize,
    key_len: usize,
    msg_ptr: usize,
    msg_len: usize,
) -> i32 {
    let key = read_bytes(key_ptr, key_len);
    let msg = read_bytes(msg_ptr, msg_len);
    match hmac_sha256_hex_core(&key, &msg) {
        Ok(hex) => set_result(hex.into_bytes()),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_ed25519_pubkey_hex(seed_ptr: usize, seed_len: usize) -> i32 {
    let seed = read_bytes(seed_ptr, seed_len);
    match ed25519_pubkey_hex_core(&seed) {
        Ok(hex) => set_result(hex.into_bytes()),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_ed25519_sign_hex(
    seed_ptr: usize,
    seed_len: usize,
    msg_ptr: usize,
    msg_len: usize,
) -> i32 {
    let seed = read_bytes(seed_ptr, seed_len);
    let msg = read_bytes(msg_ptr, msg_len);
    match ed25519_sign_hex_core(&seed, &msg) {
        Ok(hex) => set_result(hex.into_bytes()),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_x25519_pubkey_hex(sec_ptr: usize, sec_len: usize) -> i32 {
    let sec = read_bytes(sec_ptr, sec_len);
    match x25519_pubkey_hex_core(&sec) {
        Ok(hex) => set_result(hex.into_bytes()),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_build_creator_signing_message(
    hdr_sha_ptr: usize,
    hdr_sha_len: usize,
    arena_sha_ptr: usize,
    arena_sha_len: usize,
) -> i32 {
    let hdr_sha = match read_str(hdr_sha_ptr, hdr_sha_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let arena_sha = match read_str(arena_sha_ptr, arena_sha_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    match build_creator_signing_message_core(&hdr_sha, &arena_sha) {
        Ok(bytes) => set_result(bytes),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_build_marketplace_signing_message(
    hdr_sha_ptr: usize,
    hdr_sha_len: usize,
    arena_sha_ptr: usize,
    arena_sha_len: usize,
    rcpt_ptr: usize,
    rcpt_len: usize,
) -> i32 {
    let hdr_sha = match read_str(hdr_sha_ptr, hdr_sha_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let arena_sha = match read_str(arena_sha_ptr, arena_sha_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let rcpt = match read_str(rcpt_ptr, rcpt_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    match build_marketplace_signing_message_core(&hdr_sha, &arena_sha, &rcpt) {
        Ok(bytes) => set_result(bytes),
        Err(msg) => set_error(msg),
    }
}

#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_sign_creator_header(
    hdr_ptr: usize,
    hdr_len: usize,
    arena_ptr: usize,
    arena_len: usize,
    seed_ptr: usize,
    seed_len: usize,
    signer_ptr: usize,
    signer_len: usize,
    include_tier_prefixes: u8,
) -> i32 {
    let hdr = match read_str(hdr_ptr, hdr_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let arena = read_bytes(arena_ptr, arena_len);
    let seed = read_bytes(seed_ptr, seed_len);
    let signer = match read_str(signer_ptr, signer_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    match sign_creator_header_core(&hdr, &arena, &seed, &signer, include_tier_prefixes != 0) {
        Ok(json) => set_result(json.into_bytes()),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_sign_marketplace_receipt(
    hdr_ptr: usize,
    hdr_len: usize,
    arena_ptr: usize,
    arena_len: usize,
    seed_ptr: usize,
    seed_len: usize,
    spec_ptr: usize,
    spec_len: usize,
) -> i32 {
    let hdr = match read_str(hdr_ptr, hdr_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let arena = read_bytes(arena_ptr, arena_len);
    let seed = read_bytes(seed_ptr, seed_len);
    let spec = match read_str(spec_ptr, spec_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    match sign_marketplace_receipt_core(&hdr, &arena, &seed, &spec) {
        Ok(json) => set_result(json.into_bytes()),
        Err(msg) => set_error(msg),
    }
}

#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_sign_oem_device_cert(
    seed_ptr: usize,
    seed_len: usize,
    vendor_ptr: usize,
    vendor_len: usize,
    model_ptr: usize,
    model_len: usize,
    dev_id_ptr: usize,
    dev_id_len: usize,
    pk_ptr: usize,
    pk_len: usize,
) -> i32 {
    let seed = read_bytes(seed_ptr, seed_len);
    let vendor = match read_str(vendor_ptr, vendor_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let model = match read_str(model_ptr, model_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let dev_id = match read_str(dev_id_ptr, dev_id_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let pk = read_bytes(pk_ptr, pk_len);
    match sign_oem_device_cert_core(&seed, &vendor, &model, &dev_id, &pk) {
        Ok(json) => set_result(json.into_bytes()),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_verify_oem_device_cert(
    cert_ptr: usize,
    cert_len: usize,
    pk_hex_ptr: usize,
    pk_hex_len: usize,
) -> i32 {
    let cert = match read_str(cert_ptr, cert_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let pk_hex = match read_str(pk_hex_ptr, pk_hex_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    match verify_oem_device_cert_core(&cert, &pk_hex) {
        Ok(hex) => set_result(hex.into_bytes()),
        Err(msg) => set_error(msg),
    }
}

#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_build_oem_cert_v2_signing_message(
    vendor_ptr: usize,
    vendor_len: usize,
    model_ptr: usize,
    model_len: usize,
    dev_id_ptr: usize,
    dev_id_len: usize,
    pk_ptr: usize,
    pk_len: usize,
    iss_pk_ptr: usize,
    iss_pk_len: usize,
    assurance: u8,
    pol_ptr: usize,
    pol_len: usize,
    ev_ptr: usize,
    ev_len: usize,
) -> i32 {
    let vendor = match read_str(vendor_ptr, vendor_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let model = match read_str(model_ptr, model_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let dev_id = read_bytes(dev_id_ptr, dev_id_len);
    let pk = read_bytes(pk_ptr, pk_len);
    let iss_pk = read_bytes(iss_pk_ptr, iss_pk_len);
    let pol = read_opt_bytes(pol_ptr, pol_len);
    let ev = read_opt_bytes(ev_ptr, ev_len);
    match build_oem_cert_v2_signing_message_core(
        &vendor,
        &model,
        &dev_id,
        &pk,
        &iss_pk,
        assurance,
        pol.as_deref(),
        ev.as_deref(),
    ) {
        Ok(bytes) => set_result(bytes),
        Err(msg) => set_error(msg),
    }
}
