use polars::prelude::PlIndexMap;

pub type BsonDoc = bson::Document;
pub type SyncCursor = mongodb::sync::Cursor<BsonDoc>;

const DEFAULT_CHUNK_SIZE: usize = 100;

pub enum ScanStrategy {
    SingleThreadedVector,
    SingleThreadedStream,
    MultiThreaded,
}

pub fn is_whitespace(the_char: u8) -> bool {
    the_char == ' ' as u8
}

pub fn skip_whitespace(bytes: &[u8]) -> &[u8] {
    bytes.trim_ascii()
}
