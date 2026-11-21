//! `SONA` Graph Topology and Subsystem Configurations (`schema/topology.rs`).
//!
//! Canonical Reference: `docs/specs/dsp/parametric_control_bus.md` and `docs/specs/architecture/sona_architecture.md`.

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

pub const fn default_envelopes() -> u8 {
    4
}
pub const fn default_lfos() -> u8 {
    2
}
pub const fn default_lfo_harmonics() -> u8 {
    4
}
pub const fn default_fm_rank() -> u8 {
    4
}
pub const fn default_modulation_bus_dim() -> u8 {
    16
}
pub const fn default_trust_radius() -> f32 {
    2.5
}

/// Control-Bus configuration (`control_bus`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ControlBusConfig {
    /// Number of envelope follower bands (`4`: Low, Mid, High, Aux Key).
    #[serde(default = "default_envelopes")]
    pub envelopes: u8,
    /// Number of LFO oscillators (`2`).
    #[serde(default = "default_lfos")]
    pub lfo_oscillators: u8,
    /// Number of Chebyshev harmonics per LFO (`4`).
    #[serde(default = "default_lfo_harmonics")]
    pub lfo_harmonics: u8,
    /// Factorization Machine (FM) interaction rank ($R = 4$).
    #[serde(default = "default_fm_rank")]
    pub fm_interaction_rank: u8,
    /// Dimension of the unified modulation bus ($D = 16$).
    #[serde(default = "default_modulation_bus_dim")]
    pub modulation_bus_dim: u8,
    /// Trust-region soft-clip radius $M_h$ (`ref[dsp.ctrl.op-n.trust-clip]`, default `2.5`).
    #[serde(default = "default_trust_radius")]
    pub head_trust_region_radius: f32,
}

impl Default for ControlBusConfig {
    fn default() -> Self {
        Self {
            envelopes: 4,
            lfo_oscillators: 2,
            lfo_harmonics: 4,
            fm_interaction_rank: 4,
            modulation_bus_dim: 16,
            head_trust_region_radius: 2.5,
        }
    }
}

pub const fn default_fir_taps() -> u8 {
    32
}
pub const fn default_kautz_modals() -> u8 {
    8
}

/// Early FIR + Orthonormal Kautz Modal Bank convolver configuration (`modal_convolver`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ModalConvolverConfig {
    /// Number of direct early FIR taps per channel (`32, 48, 96, 128`).
    #[serde(default = "default_fir_taps")]
    pub fir_taps: u8,
    /// Number of shared Kautz modal pole pairs (`8, 12, 24, 32`).
    #[serde(default = "default_kautz_modals")]
    pub kautz_modals: u8,
}

impl Default for ModalConvolverConfig {
    fn default() -> Self {
        Self {
            fir_taps: default_fir_taps(),
            kautz_modals: default_kautz_modals(),
        }
    }
}

pub const fn default_ro_pole_48k() -> f32 {
    0.0
}

/// Learned 48 kHz linear readout leak pole configuration (`linear_readout`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LinearReadoutConfig {
    /// Learned 48 kHz readout leak pole clamped to `[0.0, dnam_kernels::MAX_PARALRU_R_F32]` (`0.0` disables `hp_lp`).
    #[serde(default = "default_ro_pole_48k")]
    pub ro_pole_48k: f32,
}

impl Default for LinearReadoutConfig {
    fn default() -> Self {
        Self {
            ro_pole_48k: default_ro_pole_48k(),
        }
    }
}

pub fn default_delay_storage() -> String {
    "fp16".to_string()
}
pub fn default_interp() -> String {
    "catmull_rom_cubic".to_string()
}
pub const fn default_tzf_tap() -> f32 {
    64.0
}
pub const fn default_switch_xfade() -> u16 {
    128
}
pub const fn default_allpass_delays() -> [u16; 16] {
    [
        19, 31, 47, 73, 103, 139, 179, 223, 29, 53, 83, 113, 151, 191, 233, 251,
    ]
}

pub fn deserialize_allpass_delays<'de, D>(deserializer: D) -> Result<[u16; 16], D::Error>
where
    D: serde::Deserializer<'de>,
{
    let v = Vec::<u16>::deserialize(deserializer)?;
    let mut out = default_allpass_delays();
    if v.len() == 8 || v.len() == 16 {
        for (i, &val) in v.iter().enumerate() {
            out[i] = val;
        }
        Ok(out)
    } else {
        Err(serde::de::Error::custom(
            "allpass_delays_samples must contain 8 or 16 u16 elements",
        ))
    }
}

pub const fn default_dc_pole() -> f32 {
    0.995
}
pub const fn default_loop_ceiling() -> f32 {
    0.985
}
/// Symmetric coercive `Op-T` Transport configuration (`transport`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TransportConfig {
    /// Delay buffer storage format (`"fp16"`).
    #[serde(default = "default_delay_storage")]
    pub delay_storage: String,
    /// Fractional delay interpolation kernel (`"catmull_rom_cubic"`).
    #[serde(default = "default_interp")]
    pub interp: String,
    /// Through-Zero Flanging (TZF) dry reference tap in samples (`64.0` at $48\text{ kHz}$).
    #[serde(default = "default_tzf_tap")]
    pub tzf_reference_tap_samples: f32,
    /// Raised-cosine switch crossfade window in samples (`128` at $48\text{ kHz}$).
    #[serde(default = "default_switch_xfade")]
    pub switch_xfade_samples: u16,
    /// Coprime Schroeder all-pass delay lengths in samples (`4..16` transport lines across `Lite`/`Full`/`Pro`/`Ultra`).
    #[serde(
        default = "default_allpass_delays",
        deserialize_with = "deserialize_allpass_delays"
    )]
    pub allpass_delays_samples: [u16; 16],
    /// In-loop $5\text{ Hz}$ DC blocker pole (`0.995` at $48\text{ kHz}$).
    #[serde(default = "default_dc_pole")]
    pub dc_blocker_pole: f32,
    /// Global loop ceiling (`0.985`).
    #[serde(default = "default_loop_ceiling")]
    pub loop_ceiling: f32,
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self {
            delay_storage: default_delay_storage(),
            interp: default_interp(),
            tzf_reference_tap_samples: 64.0,
            switch_xfade_samples: 128,
            allpass_delays_samples: default_allpass_delays(),
            dc_blocker_pole: 0.995,
            loop_ceiling: 0.985,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subsystem_configs_serde_and_defaults() {
        let bus = ControlBusConfig::default();
        assert_eq!(bus.envelopes, 4);
        assert_eq!(bus.modulation_bus_dim, 16);

        let convolver = ModalConvolverConfig::default();
        assert_eq!(convolver.fir_taps, 32);
        assert_eq!(convolver.kautz_modals, 8);

        let readout = LinearReadoutConfig::default();
        assert_eq!(readout.ro_pole_48k, 0.0);

        let transport = TransportConfig::default();
        assert_eq!(transport.delay_storage, "fp16");
        assert_eq!(transport.tzf_reference_tap_samples, 64.0);

        // Unknown fields fail serde deserialization on all subsystem configs.
        assert!(serde_json::from_str::<ModalConvolverConfig>(r#"{"shape":"bad"}"#).is_err());
        assert!(serde_json::from_str::<LinearReadoutConfig>(r#"{"extra":0.5}"#).is_err());
        assert!(serde_json::from_str::<TransportConfig>(r#"{"extra":1}"#).is_err());

        // Test allpass delays deserializer
        let json_8 = serde_json::json!([10, 20, 30, 40, 50, 60, 70, 80]);
        let parsed_8: [u16; 16] = deserialize_allpass_delays(json_8).unwrap();
        assert_eq!(parsed_8[0], 10);
        assert_eq!(parsed_8[7], 80);

        let json_invalid = serde_json::json!([1, 2, 3]);
        assert!(deserialize_allpass_delays(json_invalid).is_err());
    }
}
