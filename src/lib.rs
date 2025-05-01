pub mod buffer;
pub mod common;
pub mod from;
use std::{error::Error, ops::Deref, sync::Arc};

use buffer::{init_buffers, parse_lines};
use common::{BsonDoc, SyncCursor};
use mongodb::Cursor;
use polars::{
    error::{ErrString, PolarsError, PolarsResult},
    prelude::{AnonymousScan, Column, DataFrame, DataType, LazyFrame, Schema},
};
use polars_core::POOL;

pub struct BsonScan {
    pub cursor: Option<SyncCursor>,
    pub infer_schema_length: Option<usize>,
    pub n_threads: Option<usize>,
    pub rows_per_thread: usize,
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
    fn scan_single_threaded_stream(
        &self,
        cursor: SyncCursor,
        schema: Arc<Schema>,
    ) -> PolarsResult<DataFrame> {
        let mut buffers = init_buffers(
            self.rows_per_thread,
            schema.as_ref(),
            Some('\'' as u8),
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

#[derive(Debug)]
pub struct BsonScanOptions {
    pub infer_schema_length: Option<usize>,
}

// pub trait BsonLazyReader {
//     fn scan_bson(options: BsonScanOptions) -> PolarsResult<LazyFrame> {}
// }

// impl BsonLazyReader for LazyFrame {}
