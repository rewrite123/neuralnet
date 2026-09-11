//! GPT-2 byte-level BPE tokenizer, reading the published `vocab.json` and `merges.txt`.

use std::{collections::HashMap, fs, path::Path};

pub struct Tokenizer {
    encoder: HashMap<String, u32>,
    decoder: HashMap<u32, String>,
    ranks: HashMap<(String, String), usize>,
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
}

/// GPT-2 maps raw bytes onto printable code points so BPE can operate on text.
fn byte_char_table() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut assigned = [false; 256];
    for byte in b'!'..=b'~' { table[byte as usize] = byte as char; assigned[byte as usize] = true; }
    for byte in 0xA1u32..=0xAC { table[byte as usize] = char::from_u32(byte).unwrap(); assigned[byte as usize] = true; }
    for byte in 0xAEu32..=0xFF { table[byte as usize] = char::from_u32(byte).unwrap(); assigned[byte as usize] = true; }
    let mut extra = 0u32;
    for byte in 0..256 {
        if !assigned[byte] { table[byte] = char::from_u32(256 + extra).unwrap(); extra += 1; }
    }
    table
}

impl Tokenizer {
    pub fn load(vocab_path: &Path, merges_path: &Path) -> Result<Self, String> {
        let encoder: HashMap<String, u32> = serde_json::from_slice(&fs::read(vocab_path).map_err(|error| error.to_string())?).map_err(|error| format!("invalid vocab.json: {error}"))?;
        let merges = fs::read_to_string(merges_path).map_err(|error| error.to_string())?;
        let mut ranks = HashMap::new();
        for (rank, line) in merges.lines().filter(|line| !line.starts_with("#version") && !line.trim().is_empty()).enumerate() {
            let (left, right) = line.split_once(' ').ok_or_else(|| format!("malformed merge rule: {line}"))?;
            ranks.insert((left.to_string(), right.to_string()), rank);
        }
        let byte_to_char = byte_char_table();
        let char_to_byte = byte_to_char.iter().enumerate().map(|(byte, character)| (*character, byte as u8)).collect();
        let decoder = encoder.iter().map(|(token, id)| (*id, token.clone())).collect();
        Ok(Self { encoder, decoder, ranks, byte_to_char, char_to_byte })
    }

    pub fn vocabulary_size(&self) -> usize { self.encoder.len() }

    /// Splits text the way GPT-2's pre-tokenizer regex does, without a regex engine.
    fn pretokenize(text: &str) -> Vec<String> {
        const CONTRACTIONS: [&str; 7] = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"];
        let characters: Vec<char> = text.chars().collect();
        let mut pieces = Vec::new();
        let mut at = 0;
        while at < characters.len() {
            let remaining: String = characters[at..].iter().collect();
            if let Some(contraction) = CONTRACTIONS.iter().find(|candidate| remaining.starts_with(**candidate)) {
                pieces.push((*contraction).to_string());
                at += contraction.chars().count();
                continue;
            }
            let start = at;
            let mut cursor = at;
            if characters[cursor] == ' ' { cursor += 1; }
            if cursor < characters.len() && !characters[cursor].is_whitespace() {
                let classify = |character: char| (character.is_alphabetic(), character.is_numeric());
                let kind = classify(characters[cursor]);
                let mut end = cursor;
                while end < characters.len() && !characters[end].is_whitespace() && classify(characters[end]) == kind { end += 1; }
                pieces.push(characters[start..end].iter().collect());
                at = end;
                continue;
            }
            // A whitespace run: all but the final character when a non-space follows it.
            let mut end = at;
            while end < characters.len() && characters[end].is_whitespace() { end += 1; }
            let stop = if end < characters.len() && end - at >= 2 { end - 1 } else { end };
            pieces.push(characters[at..stop].iter().collect());
            at = stop;
        }
        pieces
    }

    fn merge(&self, piece: &str) -> Vec<String> {
        let mut symbols: Vec<String> = piece.chars().map(|character| character.to_string()).collect();
        while symbols.len() > 1 {
            let mut best: Option<(usize, usize)> = None;
            for index in 0..symbols.len() - 1 {
                if let Some(rank) = self.ranks.get(&(symbols[index].clone(), symbols[index + 1].clone())) {
                    if best.is_none_or(|(current, _)| *rank < current) { best = Some((*rank, index)); }
                }
            }
            let Some((_, index)) = best else { break };
            let (left, right) = (symbols[index].clone(), symbols[index + 1].clone());
            let mut merged = Vec::with_capacity(symbols.len());
            let mut cursor = 0;
            while cursor < symbols.len() {
                if cursor + 1 < symbols.len() && symbols[cursor] == left && symbols[cursor + 1] == right {
                    merged.push(format!("{left}{right}"));
                    cursor += 2;
                } else {
                    merged.push(symbols[cursor].clone());
                    cursor += 1;
                }
            }
            symbols = merged;
        }
        symbols
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        let mut ids = Vec::new();
        for piece in Self::pretokenize(text) {
            let mapped: String = piece.as_bytes().iter().map(|byte| self.byte_to_char[*byte as usize]).collect();
            for symbol in self.merge(&mapped) {
                ids.push(*self.encoder.get(&symbol).ok_or_else(|| format!("token {symbol:?} is not in the vocabulary"))?);
            }
        }
        Ok(ids)
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, String> {
        let mut bytes = Vec::new();
        for id in ids {
            let token = self.decoder.get(id).ok_or_else(|| format!("token id {id} is not in the vocabulary"))?;
            for character in token.chars() {
                bytes.push(*self.char_to_byte.get(&character).ok_or_else(|| format!("token {token:?} contains an unmappable character"))?);
            }
        }
        String::from_utf8(bytes).map_err(|error| error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenizer() -> Option<Tokenizer> {
        let vocab = Path::new("data/gpt2/vocab.json");
        let merges = Path::new("data/gpt2/merges.txt");
        if !vocab.exists() || !merges.exists() { return None; }
        Some(Tokenizer::load(vocab, merges).unwrap())
    }

    #[test]
    fn pretokenizer_splits_like_the_reference_regex() {
        assert_eq!(Tokenizer::pretokenize("Hello world"), vec!["Hello", " world"]);
        assert_eq!(Tokenizer::pretokenize("don't"), vec!["don", "'t"]);
        assert_eq!(Tokenizer::pretokenize("abc123"), vec!["abc", "123"]);
        assert_eq!(Tokenizer::pretokenize("a  b"), vec!["a", " ", " b"]);
        assert_eq!(Tokenizer::pretokenize("a   b"), vec!["a", "  ", " b"]);
        assert_eq!(Tokenizer::pretokenize("hi!  "), vec!["hi", "!", "  "]);
        assert_eq!(Tokenizer::pretokenize("x\ny"), vec!["x", "\n", "y"]);
    }

    #[test]
    fn encodes_known_gpt2_token_ids() {
        let Some(tokenizer) = tokenizer() else { return };
        // Ground truth taken from the published vocabulary itself, not from memory.
        assert_eq!(tokenizer.encode("Hello world").unwrap(), vec![tokenizer.encoder["Hello"], tokenizer.encoder["\u{0120}world"]]);
        assert_eq!(tokenizer.encode(" the").unwrap(), vec![tokenizer.encoder["\u{0120}the"]]);
        assert_eq!(tokenizer.encode("Hello world").unwrap(), vec![15496, 995]);
        assert_eq!(tokenizer.encode("The quick brown fox").unwrap(), vec![464, 2068, 7586, 21831]);
    }

    #[test]
    fn round_trips_text_through_encode_and_decode() {
        let Some(tokenizer) = tokenizer() else { return };
        for sample in [
            "Hello world",
            "The quick brown fox jumps over the lazy dog.",
            "don't  panic\n\nnew   paragraph\t tabbed",
            "unicode: caf\u{e9} na\u{ef}ve \u{4f60}\u{597d} \u{1f680}\u{1f680}",
            "numbers 1234567890 and symbols !@#$%^&*()",
            "   leading and trailing   ",
            "",
        ] {
            let ids = tokenizer.encode(sample).unwrap();
            assert_eq!(tokenizer.decode(&ids).unwrap(), sample, "round trip failed for {sample:?}");
        }
    }

    #[test]
    fn vocabulary_matches_the_published_size() {
        let Some(tokenizer) = tokenizer() else { return };
        assert_eq!(tokenizer.vocabulary_size(), 50257);
    }
}
