//! 与 Python 实现的互通性单测：PBKDF2 哈希逐字节一致、PyJWT 令牌可解/可验。
use crate::auth::{decode_token, hash_password, hash_token};

#[test]
fn pbkdf2_matches_python_vector() {
    // 由 Python: hashlib.pbkdf2_hmac("sha256", "P@ssw0rd-测试", salt, 210_000) 生成
    let (salt_hex, hash_hex) = hash_password(
        "P@ssw0rd-测试",
        Some("00112233445566778899aabbccddeeff"),
    );
    assert_eq!(salt_hex, "00112233445566778899aabbccddeeff");
    assert_eq!(
        hash_hex,
        "01aa368a51f3bef1c97ce71d60306bd404f4011ada5fba350092cd007000ed7b"
    );
}

#[test]
fn decodes_pyjwt_token() {
    // 由 Python PyJWT HS256 签发的固定令牌（secret=unit-test-secret, exp=2100 年）
    let token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsInVpZCI6InUxIiwicm9sZSI6ImFkbWluIiwic2lkIjoiczEiLCJ0eXAiOiJhY2Nlc3MiLCJleHAiOjQxMDI0NDQ4MDAsImlhdCI6MTc5MDAwMDAwMCwianRpIjoiajEifQ.I7licM0dJ0qbCdjAl_AoYJOAk7KBiQj8UpfSW5k961M";
    let claims = decode_token(token, "access", "unit-test-secret")
        .expect("PyJWT 令牌应可解析");
    assert_eq!(claims.sub, "admin");
    assert_eq!(claims.uid, "u1");
    assert_eq!(claims.sid, "s1");
    assert_eq!(claims.role, "admin");
    assert_eq!(claims.jti, "j1");
}

#[test]
fn rejects_wrong_type_token() {
    let token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsInVpZCI6InUxIiwicm9sZSI6ImFkbWluIiwic2lkIjoiczEiLCJ0eXAiOiJhY2Nlc3MiLCJleHAiOjQxMDI0NDQ4MDAsImlhdCI6MTc5MDAwMDAwMCwianRpIjoiajEifQ.I7licM0dJ0qbCdjAl_AoYJOAk7KBiQj8UpfSW5k961M";
    assert!(decode_token(token, "refresh", "unit-test-secret").is_err());
}

#[test]
fn token_hash_is_sha256_hex() {
    assert_eq!(
        hash_token("abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}
