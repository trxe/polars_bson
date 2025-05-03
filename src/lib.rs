pub mod buffer;
pub mod common;
pub mod from;

use std::{num::NonZeroUsize, path::PathBuf, sync::Arc};

use buffer::{infer_schema, init_buffers, parse_lines};
use common::{BsonDoc, SyncCursor};
use mongodb::{Cursor, sync::Collection};
use polars::{
    error::{PolarsError, PolarsResult},
    frame::DataFrame,
    io::{RowIndex, SerReader, mmap::MmapBytesReader, predicates::PhysicalIoExpr},
    prelude::{ArrayCollectIterExt, ArrowDataType, Column, PlSmallStr, Schema, SchemaRef},
};

pub struct BsonScan {
    pub collection: Collection<BsonDoc>,
    pub find_doc: Option<BsonDoc>,
    pub infer_schema_length: Option<usize>,
    pub n_threads: Option<usize>,
    pub ignore_errors: Option<bool>,
    pub needs_escaping: Option<bool>,
    pub allow_null: Option<bool>,
}

const DEFAULT_CHUNK_SIZE: usize = 100;

pub struct BsonReader<'a> {
    collection: mongodb::sync::Collection<BsonDoc>,
    find: BsonDoc,
    rechunk: bool,
    ignore_errors: bool,
    infer_schema_len: Option<i64>,
    batch_size: NonZeroUsize,
    projection: Option<Arc<[PlSmallStr]>>,
    schema: Option<SchemaRef>,
    schema_overwrite: Option<&'a Schema>,
    n_rows: Option<usize>,
    n_threads: Option<usize>,
    row_index: Option<&'a mut RowIndex>,
    predicate: Option<Arc<dyn PhysicalIoExpr>>,
}

impl<'a> BsonReader<'a> {
    pub fn new(collection: mongodb::sync::Collection<BsonDoc>, find: BsonDoc) -> Self {
        Self {
            collection,
            find,
            rechunk: true,
            ignore_errors: false,
            infer_schema_len: Some(100),
            batch_size: NonZeroUsize::new(8192).unwrap(),
            projection: None,
            schema: None,
            schema_overwrite: None,
            n_rows: None,
            row_index: None,
            predicate: None,
            n_threads: None,
        }
    }

    pub fn finish(self) -> PolarsResult<DataFrame> {
        let mut cursor = match self.collection.find(self.find.clone()).run() {
            Ok(x) => x,
            Err(e) => {
                return Err(PolarsError::IO {
                    error: Arc::new(MongoError(e).into()),
                    msg: Some("failed to execute query".into()),
                });
            }
        };
        let infer_len = self.infer_schema_len.unwrap_or(-1);
        let mut rows = vec![];
        loop {
            match cursor.next() {
                Some(result) => match result {
                    Ok(doc) => {
                        rows.push(doc);
                    }
                    Err(e) => {
                        return Err(PolarsError::IO {
                            error: Arc::new(MongoError(e).into()),
                            msg: Some("failed to extract bson document".into()),
                        });
                    }
                },
                None => break,
            }
        }
        let capacity = rows.len();
        let schema = infer_schema(rows.as_slice(), infer_len)?;
        let mut buffers = init_buffers(&schema, capacity, self.ignore_errors)?;
        parse_lines(rows, &mut buffers, true, true, true)?;
        let columns = buffers
            .into_iter()
            .map(|(name, buf)| buf.into_series().map(move |x| Column::new(name, x)))
            .map(|column| column.expect("Failed to parse column in mongodb document"))
            .collect::<Vec<_>>();

        DataFrame::new(columns)
    }

    pub fn with_rechunk(mut self, rechunk: bool) -> Self {
        self.rechunk = rechunk;
        self
    }

    pub fn with_n_rows(mut self, num_rows: Option<usize>) -> Self {
        self.n_rows = num_rows;
        self
    }

    /// Set the BSON file's schema
    pub fn with_schema(mut self, schema: SchemaRef) -> Self {
        self.schema = Some(schema);
        self
    }

    /// Overwrite parts of the inferred schema.
    pub fn with_schema_overwrite(mut self, schema: &'a Schema) -> Self {
        self.schema_overwrite = Some(schema);
        self
    }

    /// Infer schema length (number of records to use to infer schema from at one time)
    pub fn infer_schema_len(mut self, max_records: Option<i64>) -> Self {
        self.infer_schema_len = max_records;
        self
    }

    /// Batch size (number of records to load at one time)
    pub fn with_batch_size(mut self, batch_size: NonZeroUsize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set the reader's column projection: the names of the columns to keep after deserialization. If `None`, all
    /// columns are kept.
    ///
    /// Setting `projection` to the columns you want to keep is more efficient than deserializing all of the columns and
    /// then dropping the ones you don't want.
    pub fn with_projection(mut self, projection: Option<Arc<[PlSmallStr]>>) -> Self {
        self.projection = projection;
        self
    }

    /// Return a `null` if an error occurs during parsing.
    pub fn with_ignore_errors(mut self, ignore: bool) -> Self {
        self.ignore_errors = ignore;
        self
    }

    pub fn with_predicate(mut self, predicate: Option<Arc<dyn PhysicalIoExpr>>) -> Self {
        self.predicate = predicate;
        self
    }

    pub fn with_row_index(mut self, row_index: Option<&'a mut RowIndex>) -> Self {
        self.row_index = row_index;
        self
    }

    pub fn with_n_threads(mut self, n: Option<usize>) -> Self {
        self.n_threads = n;
        self
    }
}

struct MongoError(mongodb::error::Error);

impl From<MongoError> for std::io::Error {
    fn from(value: MongoError) -> Self {
        std::io::Error::new(std::io::ErrorKind::Interrupted, value.0)
    }
}

#[cfg(test)]
mod tests {

    use bson::doc;
    use chrono::Utc;
    use polars::prelude::LazyFrame;

    use crate::{BsonReader, common::BsonDoc};

    const MONGO_URI: &str = "mongodb://localhost:27017";
    const MONGO_DEFAULT_DB: &str = "csdb";
    #[test]
    fn valid_bigger_collection_mongo() {
        dotenvy::dotenv().unwrap();
        let modb = match std::env::var("MONGO_LIVEDB") {
            Ok(x) => x,
            Err(_) => {
                return ();
            }
        };
        let client = mongodb::sync::Client::with_uri_str(modb).expect("client not built");
        let db = client.database(MONGO_DEFAULT_DB);
        let collection = db.collection::<BsonDoc>("transactions");
        let df = BsonReader::new(collection, doc! {"team.id": 1})
            .finish()
            .unwrap();

        println!("{:?}", df);
    }

    #[test]
    fn valid_default_config_mongo() {
        let client = mongodb::sync::Client::with_uri_str(MONGO_URI).expect("client not built");
        let db = client.database(MONGO_DEFAULT_DB);
        let collection = db.collection::<BsonDoc>("persons");
        let chrono_dt: chrono::DateTime<Utc> = "2014-11-28T12:00:09Z".parse().unwrap();
        collection
            .insert_many([
                doc! { "name": "foo", "id": 1, "desc": "bar", "a_date": chrono_dt },
                doc! { "name": "dee", "id": 2, "desc": "bee" },
            ])
            .run()
            .unwrap();

        let df = BsonReader::new(collection.clone(), doc! {})
            .finish()
            .unwrap();

        println!("{:?}", df);

        collection.clone().drop().run().unwrap();
        let final_check = collection.clone().find(doc! {}).run().unwrap();
        assert!(final_check.collect::<Vec<_>>().is_empty());
    }
}
