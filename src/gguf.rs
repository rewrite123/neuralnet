//! Minimal GGUF v3 reader/writer primitives shared by the model persistence layer.

pub(crate) const ALIGNMENT: usize = 32;

pub(crate) fn aligned(value: usize) -> usize { value.next_multiple_of(ALIGNMENT) }

pub(crate) fn u32put(bytes: &mut Vec<u8>, value: u32) { bytes.extend(value.to_le_bytes()); }
pub(crate) fn u64put(bytes: &mut Vec<u8>, value: u64) { bytes.extend(value.to_le_bytes()); }
pub(crate) fn strput(bytes: &mut Vec<u8>, value: &str) { u64put(bytes, value.len() as u64); bytes.extend(value.as_bytes()); }

pub(crate) fn meta_str(bytes: &mut Vec<u8>, key: &str, value: &str) { strput(bytes, key); u32put(bytes, 8); strput(bytes, value); }
pub(crate) fn meta_u32(bytes: &mut Vec<u8>, key: &str, value: u32) { strput(bytes, key); u32put(bytes, 4); u32put(bytes, value); }
pub(crate) fn meta_u32s(bytes: &mut Vec<u8>, key: &str, values: &[u32]) { strput(bytes, key); u32put(bytes, 9); u32put(bytes, 4); u64put(bytes, values.len() as u64); for value in values { u32put(bytes, *value); } }

pub(crate) fn tensor(bytes: &mut Vec<u8>, name: &str, shape: &[u64], offset: usize) { strput(bytes, name); u32put(bytes, shape.len() as u32); for dimension in shape { u64put(bytes, *dimension); } u32put(bytes, 0); u64put(bytes, offset as u64); }
pub(crate) fn f32s(bytes: &mut Vec<u8>, values: &[f32]) { for value in values { bytes.extend(value.to_le_bytes()); } }

pub(crate) fn take<'a>(bytes: &'a [u8], at: &mut usize, count: usize) -> Result<&'a [u8], String> {
    let end = at.checked_add(count).ok_or("GGUF offset overflow")?;
    let slice = bytes.get(*at..end).ok_or("truncated GGUF")?;
    *at = end;
    Ok(slice)
}
pub(crate) fn get_u32(bytes: &[u8], at: &mut usize) -> Result<u32, String> { Ok(u32::from_le_bytes(take(bytes, at, 4)?.try_into().unwrap())) }
pub(crate) fn get_u64(bytes: &[u8], at: &mut usize) -> Result<u64, String> { Ok(u64::from_le_bytes(take(bytes, at, 8)?.try_into().unwrap())) }
pub(crate) fn get_str(bytes: &[u8], at: &mut usize) -> Result<String, String> {
    let count = get_u64(bytes, at)? as usize;
    String::from_utf8(take(bytes, at, count)?.to_vec()).map_err(|_| "invalid GGUF text".to_string())
}

/// Reads a u32 array metadata value; the element type tag has already been consumed by the caller.
pub(crate) fn get_u32_array(bytes: &[u8], at: &mut usize) -> Result<Vec<usize>, String> {
    if get_u32(bytes, at)? != 4 { return Err("expected a u32 GGUF array".into()); }
    let count = get_u64(bytes, at)? as usize;
    (0..count).map(|_| get_u32(bytes, at).map(|value| value as usize)).collect()
}

pub(crate) fn skip(bytes: &[u8], at: &mut usize, kind: u32) -> Result<(), String> {
    match kind {
        4 => { take(bytes, at, 4)?; }
        8 => { let count = get_u64(bytes, at)? as usize; take(bytes, at, count)?; }
        9 => { get_u32_array(bytes, at)?; }
        _ => return Err(format!("unsupported GGUF metadata type {kind}")),
    }
    Ok(())
}

pub(crate) fn tensor_values(bytes: &[u8], start: usize, info: &[(String, usize)], name: &str, count: usize) -> Result<Vec<f32>, String> {
    let offset = info.iter().find(|(candidate, _)| candidate == name).ok_or_else(|| format!("missing GGUF tensor {name}"))?.1;
    let data = bytes.get(start + offset..start + offset + count * 4).ok_or("truncated GGUF tensor")?;
    Ok(data.chunks_exact(4).map(|value| f32::from_le_bytes(value.try_into().unwrap())).collect())
}

/// Parses the GGUF header, returning `(metadata_keys_consumed_by, tensor_info, data_start)`.
pub(crate) fn read_header(bytes: &[u8], mut on_metadata: impl FnMut(&str, u32, &[u8], &mut usize) -> Result<bool, String>) -> Result<(Vec<(String, usize)>, usize), String> {
    let mut at = 0;
    if take(bytes, &mut at, 4)? != b"GGUF" || get_u32(bytes, &mut at)? != 3 { return Err("expected GGUF version 3".into()); }
    let tensors = get_u64(bytes, &mut at)? as usize;
    let metadata = get_u64(bytes, &mut at)? as usize;
    for _ in 0..metadata {
        let key = get_str(bytes, &mut at)?;
        let kind = get_u32(bytes, &mut at)?;
        if !on_metadata(&key, kind, bytes, &mut at)? { skip(bytes, &mut at, kind)?; }
    }
    let mut info = Vec::new();
    for _ in 0..tensors {
        let name = get_str(bytes, &mut at)?;
        let dimensions = get_u32(bytes, &mut at)? as usize;
        for _ in 0..dimensions { get_u64(bytes, &mut at)?; }
        if get_u32(bytes, &mut at)? != 0 { return Err("only F32 GGUF tensors are supported".into()); }
        info.push((name, get_u64(bytes, &mut at)? as usize));
    }
    Ok((info, aligned(at)))
}
