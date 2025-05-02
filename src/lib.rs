pub mod buffer;
pub mod common;
pub mod from;
use std::sync::Arc;

use buffer::{init_buffers, parse_lines};
use common::{BsonDoc, SyncCursor};
use from::Wrap;
use mongodb::{action::Find, sync::Collection};
use polars::{
    error::{ErrString, PolarsError, PolarsResult},
    frame::row::infer_schema,
    prelude::{
        AnonymousScan, Column, DataFrame, DataType, LazyFrame, ScanArgsAnonymous, Schema, SchemaRef,
    },
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

struct MongoError(mongodb::error::Error);

impl From<MongoError> for std::io::Error {
    fn from(value: MongoError) -> Self {
        std::io::Error::new(std::io::ErrorKind::Interrupted, value.0)
    }
}

impl BsonScan {
    fn new(
        collection: Collection<BsonDoc>,
        find_doc: Option<BsonDoc>,
        options: &BsonScanOptions,
    ) -> Self {
        Self {
            collection,
            find_doc,
            infer_schema_length: options.infer_schema_length,
            n_threads: None,
            ignore_errors: Some(false),
            needs_escaping: Some(false),
            allow_null: Some(true),
        }
    }
    fn get_cursor(&self) -> PolarsResult<SyncCursor> {
        let find_doc = self.find_doc.clone().unwrap_or_default();
        let find: Find<BsonDoc> = self.collection.find(find_doc);
        let cursor = match find.run() {
            Ok(x) => x,
            Err(e) => {
                return Err(PolarsError::IO {
                    error: Arc::new(MongoError(e).into()),
                    msg: Some(ErrString::new_static("Connection with mongodb interrupted")),
                });
            }
        };
        Ok(cursor)
    }
    fn schema_single_threaded_stream(
        &self,
        cursor: SyncCursor,
        infer_schema_length: Option<usize>,
    ) -> PolarsResult<Schema> {
        let iter = cursor.map(|doc| {
            let val = doc.unwrap();
            val.into_iter()
                .map(|(key, value)| {
                    let dtype = Wrap::<DataType>::from(&value);
                    (key, dtype.0)
                })
                .collect()
        });
        let schema = infer_schema(iter, infer_schema_length.unwrap_or(DEFAULT_CHUNK_SIZE));
        Ok(schema)
    }
    fn scan_single_threaded_stream(
        &self,
        cursor: SyncCursor,
        schema: Arc<Schema>,
        rows_per_thread: usize,
    ) -> PolarsResult<DataFrame> {
        let mut buffers = init_buffers(
            rows_per_thread,
            schema.as_ref(),
            Some(b'\''),
            polars::prelude::CsvEncoding::Utf8,
            false,
        )?;
        match parse_lines(
            cursor,
            &mut buffers,
            self.ignore_errors.unwrap_or(true),
            self.needs_escaping.unwrap_or(false),
            self.allow_null.unwrap_or(true),
        ) {
            Ok(_) => {}
            Err(e) => {
                return Err(PolarsError::IO {
                    error: Arc::new(MongoError(e).into()),
                    msg: Some(ErrString::new_static("Connection with mongodb interrupted")),
                });
            }
        };
        let series = buffers
            .into_iter()
            .map(|(name, buf)| buf.into_series().map(move |x| Column::new(name, x)))
            .map(|column| column.expect("Failed to parse column in mongodb document"))
            .collect::<Vec<_>>();
        DataFrame::new(series)
    }
}

impl AnonymousScan for BsonScan {
    fn allows_predicate_pushdown(&self) -> bool {
        false
    }
    fn allows_projection_pushdown(&self) -> bool {
        false
    }
    fn allows_slice_pushdown(&self) -> bool {
        false
    }
    fn scan(&self, scan_opts: polars::prelude::AnonymousScanArgs) -> PolarsResult<DataFrame> {
        let schema = scan_opts.schema;
        let doc_cursor = self.get_cursor()?;
        self.scan_single_threaded_stream(doc_cursor, schema.clone(), DEFAULT_CHUNK_SIZE)
    }
    fn schema(
        &self,
        _infer_schema_length: Option<usize>,
    ) -> PolarsResult<polars::prelude::SchemaRef> {
        let schema_cursor = self.get_cursor()?;
        let schema = self.schema_single_threaded_stream(schema_cursor, self.infer_schema_length)?;
        Ok(SchemaRef::new(schema))
    }

    fn next_batch(
        &self,
        _scan_opts: polars::prelude::AnonymousScanArgs,
    ) -> PolarsResult<Option<DataFrame>> {
        Ok(None)
    }

    fn as_any(&self) -> &(dyn std::any::Any + 'static) {
        todo!()
    }
}

#[derive(Debug)]
pub struct BsonScanOptions {
    pub infer_schema_length: Option<usize>,
    pub n_rows: Option<usize>,
}

impl BsonScanOptions {
    fn new(infer_schema_length: Option<usize>, n_rows: Option<usize>) -> Self {
        Self {
            infer_schema_length,
            n_rows,
        }
    }
}

pub trait MongoLazyReader {
    fn scan_mongo_collection(
        collection: Collection<BsonDoc>,
        find_doc: BsonDoc,
        options: BsonScanOptions,
    ) -> PolarsResult<LazyFrame> {
        let f = BsonScan::new(collection, Some(find_doc), &options);

        let args = ScanArgsAnonymous {
            name: "MONGO SCAN",
            infer_schema_length: options.infer_schema_length,
            n_rows: options.n_rows,
            ..ScanArgsAnonymous::default()
        };

        LazyFrame::anonymous_scan(Arc::new(f), args)
    }
}

impl MongoLazyReader for LazyFrame {}

#[cfg(test)]
mod tests {
    use bson::doc;
    use chrono::Utc;
    use polars::prelude::LazyFrame;

    use crate::{MongoLazyReader, common::BsonDoc};

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
        let collection = db.collection::<BsonDoc>("players");
        let lf = LazyFrame::scan_mongo_collection(
            collection.clone(),
            doc! {"status": "NHL", "team.id": 1},
            crate::BsonScanOptions::new(None, None),
        );
        let df = lf.unwrap().collect().unwrap();

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

        let lf = LazyFrame::scan_mongo_collection(
            collection.clone(),
            doc! {},
            crate::BsonScanOptions::new(None, None),
        );

        let df = lf.unwrap().collect().unwrap();

        println!("{:?}", df);

        collection.clone().drop().run().unwrap();
        let final_check = collection.clone().find(doc! {}).run().unwrap();
        assert!(final_check.collect::<Vec<_>>().is_empty());
    }
}
