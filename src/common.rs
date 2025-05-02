pub type BsonDoc = bson::Document;
pub type SyncCursor = mongodb::sync::Cursor<BsonDoc>;

pub enum ScanStrategy {
    SingleThreadedVector,
    SingleThreadedStream,
    MultiThreaded,
}

pub fn is_whitespace(the_char: u8) -> bool {
    the_char == b' '
}

pub fn skip_whitespace(bytes: &[u8]) -> &[u8] {
    bytes.trim_ascii()
}
