//! Rust tokenizers crate 与 HF Python tokenizer 的 id 级对拍。

use std::path::Path;

use serde::Deserialize;
use tokenizers::Tokenizer;

#[derive(Deserialize)]
struct IdsGolden {
    instruction: String,
    queries: Vec<String>,
    ids: Vec<Vec<u32>>,
}

#[test]
fn tokenizer_ids_match_hf() {
    let fixture_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tokenizer_ids.json");
    if !fixture_path.exists() {
        eprintln!("跳过: id 夹具不存在");
        return;
    }
    let golden: IdsGolden =
        serde_json::from_reader(std::fs::File::open(&fixture_path).unwrap()).unwrap();
    let tokenizer_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../backend/app/models/bge-base-zh-v1.5/tokenizer.json");
    if !tokenizer_path.exists() {
        eprintln!("跳过: tokenizer.json 不存在");
        return;
    }
    let tokenizer = Tokenizer::from_file(&tokenizer_path).unwrap();

    for (query, expected) in golden.queries.iter().zip(&golden.ids) {
        let full = format!("{}{}", golden.instruction, query);
        let encoding = tokenizer.encode(full.as_str(), true).unwrap();
        let actual: Vec<u32> = encoding.get_ids().to_vec();
        if actual != *expected {
            // 定位首个分歧
            let mut position = 0;
            for (index, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
                if a != b {
                    position = index;
                    break;
                }
            }
            panic!(
                "id 不一致 query={:?}\n first_diff@{} actual={:?} expected={:?}\n full_actual={:?}",
                query,
                position,
                &actual[position.saturating_sub(2)..(position + 4).min(actual.len())],
                &expected[position.saturating_sub(2)..(position + 4).min(expected.len())],
                actual
            );
        }
    }
}
