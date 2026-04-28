#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use fsci_io::wav_read;
use libfuzzer_sys::fuzz_target;

const MAX_RAW_INPUT: usize = 8192;
const MAX_CHUNK_PAYLOAD: usize = 2048;

#[derive(Debug, Arbitrary)]
struct WavSpec {
    audio_format_hint: u8,
    channel_hint: u8,
    sample_rate_hint: u32,
    bit_depth_hint: u8,
    payload: Vec<u8>,
    junk_payload: Vec<u8>,
    chunk_order_hint: u8,
    truncate_tail_hint: u8,
}

fn choose_audio_format(raw: u8) -> u16 {
    match raw % 5 {
        0 => 1,
        1 => 3,
        2 => 0,
        3 => 6,
        _ => 65_534,
    }
}

fn choose_bits_per_sample(raw: u8) -> u16 {
    match raw % 8 {
        0 => 8,
        1 => 16,
        2 => 24,
        3 => 32,
        4 => 0,
        5 => 7,
        6 => 20,
        _ => 64,
    }
}

fn append_chunk(buf: &mut Vec<u8>, id: &[u8; 4], data: &[u8]) {
    buf.extend_from_slice(id);
    buf.extend_from_slice(&(data.len() as u32).to_le_bytes());
    buf.extend_from_slice(data);
    if !data.len().is_multiple_of(2) {
        buf.push(0);
    }
}

fn make_fmt_chunk(spec: &WavSpec) -> Vec<u8> {
    let audio_format = choose_audio_format(spec.audio_format_hint);
    let channels = u16::from(spec.channel_hint % 9);
    let sample_rate = spec.sample_rate_hint % 384_001;
    let bits_per_sample = choose_bits_per_sample(spec.bit_depth_hint);
    let block_align = channels.saturating_mul((bits_per_sample / 8).max(1));
    let byte_rate = sample_rate.saturating_mul(u32::from(block_align));

    let mut chunk = Vec::with_capacity(16);
    chunk.extend_from_slice(&audio_format.to_le_bytes());
    chunk.extend_from_slice(&channels.to_le_bytes());
    chunk.extend_from_slice(&sample_rate.to_le_bytes());
    chunk.extend_from_slice(&byte_rate.to_le_bytes());
    chunk.extend_from_slice(&block_align.to_le_bytes());
    chunk.extend_from_slice(&bits_per_sample.to_le_bytes());
    chunk
}

fn structured_wav_bytes(spec: &WavSpec) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(b"WAVE");

    let fmt = make_fmt_chunk(spec);
    let payload_len = spec.payload.len().min(MAX_CHUNK_PAYLOAD);
    let junk_len = spec.junk_payload.len().min(MAX_CHUNK_PAYLOAD);
    let payload = &spec.payload[..payload_len];
    let junk = &spec.junk_payload[..junk_len];

    match spec.chunk_order_hint % 4 {
        0 => {
            append_chunk(&mut buf, b"fmt ", &fmt);
            append_chunk(&mut buf, b"data", payload);
        }
        1 => {
            append_chunk(&mut buf, b"JUNK", junk);
            append_chunk(&mut buf, b"fmt ", &fmt);
            append_chunk(&mut buf, b"data", payload);
        }
        2 => {
            append_chunk(&mut buf, b"data", payload);
            append_chunk(&mut buf, b"fmt ", &fmt);
        }
        _ => {
            append_chunk(&mut buf, b"fmt ", &fmt);
            append_chunk(&mut buf, b"JUNK", junk);
            append_chunk(&mut buf, b"data", payload);
        }
    }

    let riff_size = u32::try_from(buf.len().saturating_sub(8)).unwrap_or(u32::MAX);
    buf[4..8].copy_from_slice(&riff_size.to_le_bytes());

    let truncate_tail = usize::from(spec.truncate_tail_hint % 9);
    if truncate_tail > 0 && truncate_tail < buf.len() {
        buf.truncate(buf.len() - truncate_tail);
    }

    buf
}

fn exercise_wav(bytes: &[u8]) {
    if let Ok(wav) = wav_read(bytes) {
        assert!(wav.sample_rate > 0);
        assert!(wav.channels > 0);
        assert!(matches!(wav.bits_per_sample, 8 | 16 | 24 | 32));
        assert!(wav.data.len().is_multiple_of(usize::from(wav.channels)));
    }
}

fuzz_target!(|bytes: &[u8]| {
    if bytes.len() > MAX_RAW_INPUT {
        return;
    }

    exercise_wav(bytes);

    let mut unstructured = Unstructured::new(bytes);
    if let Ok(spec) = WavSpec::arbitrary(&mut unstructured) {
        let structured = structured_wav_bytes(&spec);
        exercise_wav(&structured);
    }
});
