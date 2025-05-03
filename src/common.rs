pub type BsonDoc = bson::Document;
pub type SyncCursor = mongodb::sync::Cursor<BsonDoc>;

pub enum ScanStrategy {
    SingleThreadedVector,
    SingleThreadedStream,
    MultiThreaded,
}
