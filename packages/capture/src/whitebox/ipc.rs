#![forbid(unsafe_code)]
//! `DNWB` Stem Response Framing, Single-Request Pipe Server & Subprocess Executor (`whitebox::ipc`).

mod response;

pub use response::{
    read_dnwb_stem_response, write_dnwb_postcard_stem_response, write_dnwb_stem_response,
};

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::time::Duration;

use super::{DnwbCommand, DnwbError, DnwbWorkerSession, read_dnwb_stimulus_request_with_switches};
use crate::targets::{DistillationTarget, StemResponse, StemStimulus, StimulusFrame};

#[allow(clippy::too_many_arguments)]
pub(super) fn validate_runner_stimulus_layout(
    sample_rate_hz: u32,
    in_1: &[f32],
    in_2: Option<&[f32]>,
    in_3: Option<&[f32]>,
    in_4: Option<&[f32]>,
    control_values: &[f32],
    control_trajectories: Option<&Vec<Vec<f32>>>,
    freeze_window: Option<[usize; 2]>,
) -> Result<(), DnwbError> {
    if !(8_000..=1_536_000).contains(&sample_rate_hz) {
        return Err(DnwbError::InvalidStimulusLayout(format!(
            "sample_rate_hz {sample_rate_hz} outside supported [8000, 1536000]"
        )));
    }
    let n_frames = in_1.len();
    for (ch_num, opt) in [(2usize, in_2), (3, in_3), (4, in_4)] {
        if let Some(ch) = opt
            && ch.len() != n_frames
        {
            return Err(DnwbError::InvalidStimulusLayout(format!(
                "input channel {ch_num} length {} != in_1 length {n_frames}",
                ch.len()
            )));
        }
    }
    if let Some(trajs) = control_trajectories {
        if trajs.len() > u16::MAX as usize || trajs.len() != control_values.len() {
            return Err(DnwbError::InvalidStimulusLayout(format!(
                "control_trajectories len {} != control_values len {}",
                trajs.len(),
                control_values.len()
            )));
        }
        for (p, t) in trajs.iter().enumerate() {
            if t.len() != n_frames {
                return Err(DnwbError::InvalidStimulusLayout(format!(
                    "control_trajectories[{p}] len {} != num_frames {n_frames}",
                    t.len()
                )));
            }
        }
    }
    if let Some([start, end]) = freeze_window
        && (start > end || end > n_frames)
    {
        return Err(DnwbError::InvalidStimulusLayout(format!(
            "invalid freeze_transition_window [{start}, {end}] for num_frames {n_frames}"
        )));
    }
    Ok(())
}

/// Serves a single `DNWB` pipe request from `reader`, invoking `on_switches` before dispatching
/// to `target` and writing the `DNWB` response to `writer`.
pub fn serve_dnwb_single_request_with_switches<R, W, T, F>(
    reader: &mut R,
    writer: &mut W,
    target: &mut T,
    on_switches: F,
) -> Result<(), DnwbError>
where
    R: Read,
    W: Write,
    T: DistillationTarget,
    F: FnOnce(&mut T, &[u8]),
{
    let (hdr, stim, switches) = read_dnwb_stimulus_request_with_switches(reader)?;
    on_switches(target, &switches);
    let resp = match hdr.command {
        DnwbCommand::Reset => {
            target.reset_state();
            StemResponse {
                stem_id: stim.stem_id,
                sample_rate_hz: stim.sample_rate_hz,
                out_1: Vec::new(),
                out_2: None,
                out_3: None,
                out_4: None,
                internal_nodes: BTreeMap::new(),
            }
        }
        DnwbCommand::RenderStem => target
            .render_stem(&stim)
            .map_err(|e| DnwbError::IoFailure(format!("{e:?}")))?,
        DnwbCommand::RenderFrame => {
            let req_stem_id = stim.stem_id;
            let frame_index = req_stem_id
                .strip_prefix("frame_")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0);
            let frame = StimulusFrame {
                frame_index,
                sample_rate_hz: stim.sample_rate_hz,
                in_1: stim.in_1,
                in_2: stim.in_2,
                in_3: stim.in_3,
                in_4: stim.in_4,
                control_values: stim.control_values,
                control_trajectories: stim.control_trajectories,
            };
            let mut r = target
                .render_frame(&frame)
                .map_err(|e| DnwbError::IoFailure(format!("{e:?}")))?;
            r.stem_id = req_stem_id;
            r
        }
    };
    write_dnwb_stem_response(writer, hdr.command, &resp)?;
    writer
        .flush()
        .map_err(|e| DnwbError::IoFailure(e.to_string()))?;
    Ok(())
}

/// Serves a single `DNWB` pipe request from `reader`, dispatching to `target` and writing the
/// `DNWB` response to `writer`.
pub fn serve_dnwb_single_request<R: Read, W: Write, T: DistillationTarget>(
    reader: &mut R,
    writer: &mut W,
    target: &mut T,
) -> Result<(), DnwbError> {
    serve_dnwb_single_request_with_switches(reader, writer, target, |_, _| {})
}

/// Executes an out-of-process GPL-isolated worker command (`cmd[0] cmd[1..]`) over `stdin`/`stdout`
/// using `64 B` `DNWB` pipe framing with concurrent non-deadlocking `stdin`/`stdout` pump threads
/// and a wall-clock watchdog `timeout`.
pub fn execute_dnwb_subprocess_request(
    cmd: &[String],
    command: DnwbCommand,
    stimulus: &StemStimulus,
    switches: &[u8],
    timeout: Duration,
) -> Result<StemResponse, DnwbError> {
    let mut session = DnwbWorkerSession::new();
    session.execute(cmd, command, stimulus, switches, timeout)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::whitebox::{
        DNWB_CHANNEL_MASK_SHIFT, DNWB_FLAG_STRUCTURED_V1, FaustRunner, SpiceRunner,
        write_dnwb_stimulus_request, write_dnwb_stimulus_request_with_switches,
    };
    use std::io::Cursor;
    use std::time::Instant;

    #[test]
    fn test_ipc_response_guards_and_server_error_propagation() {
        let empty_resp = StemResponse {
            stem_id: String::from("r0"),
            sample_rate_hz: 48_000,
            out_1: Vec::new(),
            out_2: None,
            out_3: None,
            out_4: None,
            internal_nodes: BTreeMap::new(),
        };
        let mut wire = Vec::new();
        let hdr = write_dnwb_stem_response(&mut wire, DnwbCommand::Reset, &empty_resp).unwrap();

        let mut zero_ch_hdr = hdr;
        zero_ch_hdr.num_output_channels = 0;
        zero_ch_hdr.flags = DNWB_FLAG_STRUCTURED_V1;
        let mut zero_ch_wire = Vec::from(zero_ch_hdr.to_bytes());
        zero_ch_wire.extend_from_slice(&wire[64..]);
        let (_, z_resp) = read_dnwb_stem_response(&mut Cursor::new(zero_ch_wire)).unwrap();
        assert!(z_resp.out_1.is_empty());

        let mut unflagged_hdr = hdr;
        unflagged_hdr.flags = 0;
        assert!(matches!(
            read_dnwb_stem_response(&mut Cursor::new(unflagged_hdr.to_bytes())),
            Err(DnwbError::InvalidStimulusLayout(_))
        ));

        let mono_resp = StemResponse {
            stem_id: String::from("mono_only"),
            sample_rate_hz: 48_000,
            out_1: vec![0.25, -0.5],
            out_2: None,
            out_3: None,
            out_4: None,
            internal_nodes: BTreeMap::new(),
        };
        let mut mono_wire = Vec::new();
        write_dnwb_stem_response(&mut mono_wire, DnwbCommand::RenderStem, &mono_resp).unwrap();
        let (_, mono_rt) = read_dnwb_stem_response(&mut Cursor::new(mono_wire)).unwrap();
        assert_eq!(mono_rt, mono_resp);

        let mut huge_id_resp = empty_resp.clone();
        huge_id_resp.stem_id = "x".repeat(usize::from(u16::MAX) + 1);
        assert!(
            write_dnwb_stem_response(&mut Vec::new(), DnwbCommand::Reset, &huge_id_resp).is_err()
        );
        let mut huge_node_resp = empty_resp;
        huge_node_resp
            .internal_nodes
            .insert("n".repeat(usize::from(u16::MAX) + 1), vec![0.0]);
        assert!(
            write_dnwb_stem_response(&mut Vec::new(), DnwbCommand::Reset, &huge_node_resp).is_err()
        );
        let mut bad_utf8_id = zero_ch_hdr;
        bad_utf8_id.payload_byte_len = 4;
        let mut bad_utf8_wire = Vec::from(bad_utf8_id.to_bytes());
        bad_utf8_wire.extend_from_slice(&2u16.to_le_bytes());
        bad_utf8_wire.extend_from_slice(&[0xff, 0xff]);
        assert!(read_dnwb_stem_response(&mut Cursor::new(bad_utf8_wire)).is_err());

        let faust_json = r#"{"ui":[{"items":[{"type":"hslider","label":"Drive","init":0.5,"min":0.0,"max":1.0,"step":0.01},{"type":"checkbox","label":"Bright"}]}]}"#;
        let mut faust = FaustRunner::from_faust_json("ipc.dsp", faust_json, 2_000).unwrap();
        let ok_stim = StemStimulus {
            stem_id: String::from("ok_faust"),
            sample_rate_hz: 48_000,
            in_1: vec![0.2, -0.2],
            in_2: None,
            in_3: None,
            in_4: None,
            control_values: vec![0.3],
            control_trajectories: None,
            freeze_transition_window: None,
        };
        for cmd in [
            DnwbCommand::Reset,
            DnwbCommand::RenderStem,
            DnwbCommand::RenderFrame,
        ] {
            let mut req = Vec::new();
            write_dnwb_stimulus_request_with_switches(&mut req, cmd, &ok_stim, &[1]).unwrap();
            let mut out = Vec::new();
            serve_dnwb_single_request_with_switches(
                &mut Cursor::new(req),
                &mut out,
                &mut faust,
                |t, sw| t.set_switches(sw),
            )
            .unwrap();
            let (_, r) = read_dnwb_stem_response(&mut Cursor::new(out)).unwrap();
            if cmd != DnwbCommand::Reset {
                assert_eq!(r.out_1.len(), 2);
            }
        }

        let mut bad_mask_wire = wire;
        bad_mask_wire[18..20].copy_from_slice(&3u16.to_le_bytes());
        assert!(matches!(
            read_dnwb_stem_response(&mut Cursor::new(bad_mask_wire)),
            Err(DnwbError::InvalidStimulusLayout(_))
        ));
        let mut no_out1_hdr = hdr;
        no_out1_hdr.num_output_channels = 1;
        no_out1_hdr.flags = DNWB_FLAG_STRUCTURED_V1 | (0b0010 << DNWB_CHANNEL_MASK_SHIFT);
        let mut no_out1_wire = Vec::from(no_out1_hdr.to_bytes());
        no_out1_wire.extend_from_slice(&[0u8; 4]);
        assert!(matches!(
            read_dnwb_stem_response(&mut Cursor::new(no_out1_wire)),
            Err(DnwbError::InvalidStimulusLayout(_))
        ));

        let mut failing_spice = SpiceRunner::new("timeout.cir", 0);
        for cmd in [DnwbCommand::RenderStem, DnwbCommand::RenderFrame] {
            let mut req = Vec::new();
            write_dnwb_stimulus_request(&mut req, cmd, &ok_stim).unwrap();
            assert!(
                serve_dnwb_single_request(
                    &mut Cursor::new(req),
                    &mut Vec::new(),
                    &mut failing_spice
                )
                .is_err()
            );
        }

        assert!(
            validate_runner_stimulus_layout(4_000, &[0.0], None, None, None, &[], None, None)
                .is_err()
        );
        assert!(matches!(
            execute_dnwb_subprocess_request(
                &[String::from("sh")],
                DnwbCommand::Reset,
                &ok_stim,
                &[],
                Duration::ZERO,
            ),
            Err(DnwbError::WatchdogTimeout { timeout_ms: 0 })
        ));
        assert!(
            execute_dnwb_subprocess_request(
                &[],
                DnwbCommand::Reset,
                &ok_stim,
                &[],
                Duration::from_millis(50),
            )
            .is_err()
        );
        #[cfg(unix)]
        {
            let t0 = Instant::now();
            assert!(matches!(
                execute_dnwb_subprocess_request(
                    &[
                        String::from("sh"),
                        String::from("-c"),
                        String::from("sleep 2 & sleep 2"),
                    ],
                    DnwbCommand::Reset,
                    &ok_stim,
                    &[],
                    Duration::from_millis(25),
                ),
                Err(DnwbError::WatchdogTimeout { .. })
            ));
            assert!(t0.elapsed() < Duration::from_millis(500));
        }

        let mut internal_nodes = BTreeMap::new();
        internal_nodes.insert(String::from("node_a"), vec![0.25, -0.25, 0.5, -0.5]);
        let resp = StemResponse {
            stem_id: String::from("stem_4ch_96k"),
            sample_rate_hz: 96_000,
            out_1: vec![0.2, -0.4, 0.6, -0.8],
            out_2: Some(vec![-0.2, 0.4, -0.6, 0.8]),
            out_3: Some(vec![0.1, 0.1, -0.1, -0.1]),
            out_4: Some(vec![-0.1, -0.1, 0.1, 0.1]),
            internal_nodes,
        };
        let mut resp_wire = Vec::new();
        let resp_hdr =
            write_dnwb_stem_response(&mut resp_wire, DnwbCommand::RenderStem, &resp).unwrap();
        let (resp_hdr_rt, resp_rt) =
            read_dnwb_stem_response(&mut Cursor::new(resp_wire.clone())).unwrap();
        assert_eq!(resp_hdr, resp_hdr_rt);
        assert_eq!(resp_rt, resp);

        let mut nan_wire = resp_wire.clone();
        let first_sample_off = 64 + 2 + resp.stem_id.len();
        nan_wire[first_sample_off..first_sample_off + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(read_dnwb_stem_response(&mut Cursor::new(nan_wire)).is_err());
        assert!(read_dnwb_stem_response(&mut Cursor::new(&resp_wire[..68])).is_err());
    }
}
