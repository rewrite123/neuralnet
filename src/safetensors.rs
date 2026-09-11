//! Minimal safetensors reader, used to import pretrained embedding matrices.

use std::{collections::BTreeMap, fs, path::Path};

pub struct TensorEntry {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub start: usize,
    pub end: usize,
}

pub fn index(bytes: &[u8]) -> Result<(BTreeMap<String, TensorEntry>, usize), String> {
    if bytes.len() < 8 { return Err("file is too small to be safetensors".into()); }
    let header_length = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header_end = 8usize.checked_add(header_length).ok_or("safetensors header length overflows")?;
    if header_end > bytes.len() { return Err("safetensors header extends past the end of the file".into()); }
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..header_end]).map_err(|error| format!("invalid safetensors header: {error}"))?;
    let object = header.as_object().ok_or("safetensors header must be a JSON object")?;
    let mut entries = BTreeMap::new();
    for (name, value) in object {
        if name == "__metadata__" { continue; }
        let dtype = value.get("dtype").and_then(|value| value.as_str()).ok_or("tensor entry lacks dtype")?.to_string();
        let shape: Vec<usize> = value.get("shape").and_then(|value| value.as_array()).ok_or("tensor entry lacks shape")?
            .iter().map(|value| value.as_u64().map(|value| value as usize).ok_or_else(|| "shape entries must be integers".to_string())).collect::<Result<_, _>>()?;
        let offsets = value.get("data_offsets").and_then(|value| value.as_array()).ok_or("tensor entry lacks data_offsets")?;
        if offsets.len() != 2 { return Err("data_offsets must have two entries".into()); }
        let start = offsets[0].as_u64().ok_or("bad data offset")? as usize;
        let end = offsets[1].as_u64().ok_or("bad data offset")? as usize;
        entries.insert(name.clone(), TensorEntry { dtype, shape, start, end });
    }
    Ok((entries, header_end))
}

/// Reads one tensor as f32, converting from F16 or BF16 when needed.
pub fn read_tensor(path: &Path, name: &str) -> Result<(Vec<usize>, Vec<f32>), String> {
    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    let (entries, data_start) = index(&bytes)?;
    let entry = entries.get(name).ok_or_else(|| format!("safetensors file has no tensor named {name}; available: {}", entries.keys().take(20).cloned().collect::<Vec<_>>().join(", ")))?;
    let start = data_start + entry.start;
    let end = data_start + entry.end;
    let raw = bytes.get(start..end).ok_or("tensor data extends past the end of the file")?;
    let expected: usize = entry.shape.iter().product();
    let values = match entry.dtype.as_str() {
        "F32" => raw.chunks_exact(4).map(|value| f32::from_le_bytes(value.try_into().unwrap())).collect::<Vec<_>>(),
        "F16" => raw.chunks_exact(2).map(|value| f16_to_f32(u16::from_le_bytes(value.try_into().unwrap()))).collect(),
        "BF16" => raw.chunks_exact(2).map(|value| f32::from_bits((u16::from_le_bytes(value.try_into().unwrap()) as u32) << 16)).collect(),
        other => return Err(format!("unsupported safetensors dtype {other}")),
    };
    if values.len() != expected { return Err(format!("tensor {name} has {} values but its shape implies {expected}", values.len())); }
    Ok((entry.shape.clone(), values))
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exponent = ((bits >> 10) & 0x1f) as u32;
    let mantissa = (bits & 0x3ff) as u32;
    match exponent {
        0 if mantissa == 0 => f32::from_bits(sign),
        0 => {
            // Subnormal: the value is mantissa * 2^-24, renormalised into f32's exponent range.
            let position = 31 - mantissa.leading_zeros();
            f32::from_bits(sign | ((103 + position) << 23) | ((mantissa << (23 - position)) & 0x7f_ffff))
        }
        0x1f => f32::from_bits(sign | 0x7f80_0000 | (mantissa << 13)),
        _ => f32::from_bits(sign | ((exponent + 127 - 15) << 23) | (mantissa << 13)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(dtype: &str, shape: &[usize], payload: &[u8]) -> Vec<u8> {
        let header = format!("{{\"t\":{{\"dtype\":\"{dtype}\",\"shape\":{shape:?},\"data_offsets\":[0,{}]}}}}", payload.len());
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header.as_bytes());
        bytes.extend(payload);
        bytes
    }

    #[test]
    fn reads_f32_bf16_and_f16_tensors() {
        let directory = std::env::temp_dir();

        let f32_payload: Vec<u8> = [1.5f32, -2.25, 0.0, 7.5].iter().flat_map(|value| value.to_le_bytes()).collect();
        let path = directory.join("neuralnet-st-f32.safetensors");
        fs::write(&path, build("F32", &[2, 2], &f32_payload)).unwrap();
        let (shape, values) = read_tensor(&path, "t").unwrap();
        assert_eq!(shape, vec![2, 2]);
        assert_eq!(values, vec![1.5, -2.25, 0.0, 7.5]);
        fs::remove_file(&path).unwrap();

        // BF16 is the top 16 bits of the f32 pattern, so these round-trip exactly.
        let bf16_payload: Vec<u8> = [1.5f32, -2.25].iter().flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes()).collect();
        let path = directory.join("neuralnet-st-bf16.safetensors");
        fs::write(&path, build("BF16", &[2], &bf16_payload)).unwrap();
        assert_eq!(read_tensor(&path, "t").unwrap().1, vec![1.5, -2.25]);
        fs::remove_file(&path).unwrap();

        // F16: 1.5 = 0x3E00, -2.25 = 0xC080, smallest subnormal = 0x0001, largest subnormal = 0x03FF.
        let path = directory.join("neuralnet-st-f16.safetensors");
        fs::write(&path, build("F16", &[4], &[0x00, 0x3E, 0x80, 0xC0, 0x01, 0x00, 0xFF, 0x03])).unwrap();
        let values = read_tensor(&path, "t").unwrap().1;
        assert_eq!(values[0], 1.5);
        assert_eq!(values[1], -2.25);
        assert_eq!(values[2], 2.0f32.powi(-24), "smallest subnormal decoded as {}", values[2]);
        assert_eq!(values[3], 1023.0 * 2.0f32.powi(-24), "largest subnormal decoded as {}", values[3]);
        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn reports_missing_tensors_and_bad_headers() {
        let path = std::env::temp_dir().join("neuralnet-st-missing.safetensors");
        fs::write(&path, build("F32", &[1], &1.0f32.to_le_bytes())).unwrap();
        assert!(read_tensor(&path, "absent").unwrap_err().contains("no tensor named absent"));
        fs::remove_file(&path).unwrap();
        assert!(index(&[0, 0, 0]).is_err());
        assert!(index(&u64::MAX.to_le_bytes()).is_err());
    }
}
