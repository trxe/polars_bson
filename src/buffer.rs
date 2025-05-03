use std::{collections::HashMap, num::NonZeroUsize, sync::Arc};

use bson::{Binary, Bson, DateTime};
use polars::{
    error::{PolarsError, PolarsResult, polars_bail, polars_err},
    frame::row::AnyValueBuffer,
    prelude::{
        AnyValue, ArrowDataType, ChunkedBuilder, CompatLevel, DataType, DateType, Field,
        PlIndexMap, PlSmallStr, PolarsNumericType, Schema, TimeUnit, dtype_col,
    },
    series::Series,
};
use polars_core::utils::{arrow::array::StructArray, dtypes_to_supertype};
use polars_time::prelude::string::infer::{DatetimeInfer, TryFromWithUnit};

use crate::{
    common::BsonDoc,
    from::{Wrap, coerce_dtype_arrow},
};

/// Infers the [`ArrowDataType`] from an NDJSON file, optionally only using `number_of_rows` rows.
///
/// # Implementation
/// This implementation reads the file line by line and infers the type of each line.
/// It performs both `O(N)` IO and CPU-bounded operations where `N` is the number of rows.
pub fn iter_unique_dtypes(
    documents: &[BsonDoc],
    number_of_rows: Option<usize>,
) -> PolarsResult<impl Iterator<Item = ArrowDataType>> {
    if documents.is_empty() {
        return Err(PolarsError::ComputeError(
            "No results found in returned Bson document array".into(),
        ));
    }
    let doc_len = documents.len();
    let mut dtypes: Vec<Wrap<ArrowDataType>> = vec![];
    for row in 0..number_of_rows.unwrap_or(doc_len) {
        if row > doc_len {
            break;
        }
        match documents.get(row) {
            Some(doc) => dtypes.push(doc.into()),
            None => break,
        };
    }

    Ok(dtypes.into_iter().map(|x| x.0))
}

pub fn infer_schema(documents: &[BsonDoc], infer_schema_len: i64) -> PolarsResult<Schema> {
    let arrow_dtypes = iter_unique_dtypes(
        documents,
        if infer_schema_len < 0 {
            None
        } else {
            Some(infer_schema_len as usize)
        },
    )?;
    let allow_dtypes = arrow_dtypes
        .map(|dt| DataType::from_arrow_dtype(&dt))
        .collect::<Vec<_>>();
    let dtype = dtypes_to_supertype(allow_dtypes.iter())?;
    let schema = StructArray::get_fields(&dtype.to_arrow(CompatLevel::newest()))
        .iter()
        .map(Into::<Field>::into)
        .collect();
    Ok(schema)
}

pub struct DataBuffer<'a> {
    name: &'a str,
    ignore_errors: bool,
    buf: AnyValueBuffer<'a>,
}

impl DataBuffer<'_> {
    pub fn into_series(self) -> PolarsResult<Series> {
        let mut buf = self.buf;
        let mut s = buf.reset(0);
        s.rename(PlSmallStr::from_str(self.name));
        Ok(s)
    }

    #[inline]
    pub fn add(&mut self, bson: &Bson) -> PolarsResult<()> {
        use AnyValueBuffer::*;
        match &mut self.buf {
            Boolean(buf) => {
                match bson.as_bool() {
                    Some(val) => buf.append_value(val),
                    None => buf.append_null(),
                }
                Ok(())
            }
            Int32(buf) => {
                match bson.as_i32() {
                    Some(val) => buf.append_value(val),
                    None => buf.append_null(),
                }
                Ok(())
            }
            Int64(buf) => {
                match bson.as_i64() {
                    Some(val) => buf.append_value(val),
                    None => buf.append_null(),
                }
                Ok(())
            }
            UInt64(buf) => {
                match bson.as_i64() {
                    Some(val) => buf.append_value(val as u64),
                    None => buf.append_null(),
                }
                Ok(())
            }
            UInt32(buf) => {
                match bson.as_i64() {
                    Some(val) => buf.append_value(val as u32),
                    None => buf.append_null(),
                }
                Ok(())
            }
            Float32(buf) => {
                match bson.as_f64() {
                    Some(val) => buf.append_value(val as f32),
                    None => buf.append_null(),
                }
                Ok(())
            }
            Float64(buf) => {
                match bson.as_f64() {
                    Some(val) => buf.append_value(val),
                    None => buf.append_null(),
                }
                Ok(())
            }

            String(buf) => {
                match bson.as_str() {
                    Some(val) => buf.append_value(val),
                    None => {
                        let p = bson.to_string();
                        if p.is_empty() {
                            buf.append_null();
                        } else {
                            buf.append_value(p);
                        }
                    }
                }
                Ok(())
            }
            Datetime(buf, tu, _) => {
                match bson.as_datetime() {
                    Some(val) => buf.append_value(datetime_to_time_since_epoch(val, *tu)),
                    None => buf.append_null(),
                }
                Ok(())
            }
            Date(buf) => {
                match bson.as_datetime() {
                    Some(val) => buf.append_value(datetime_to_days_since_epoch(val)),
                    None => buf.append_null(),
                }
                Ok(())
            }
            All(dtype, buf) => {
                let av = deserialize_all(bson, dtype, self.ignore_errors)?;
                buf.push(av);
                Ok(())
            }
            Null(builder) => {
                match bson.as_null() {
                    Some(()) => builder.append_null(),
                    None => {
                        polars_bail!(ComputeError: "got non-null value for NULL-typed column: {}", bson)
                    }
                }
                Ok(())
            }
            _ => panic!("unexpected dtype when deserializing ndjson"),
        }
    }

    pub fn add_null(&mut self) {
        self.buf.add(AnyValue::Null).expect("should not fail");
    }
}

const MS_PER_DAY: i64 = 1000 * 60 * 60 * 24;

fn datetime_to_days_since_epoch(val: &DateTime) -> i32 {
    let ms = val.timestamp_millis();
    return (ms / MS_PER_DAY) as i32;
}

fn datetime_to_time_since_epoch(val: &DateTime, tu: TimeUnit) -> i64 {
    let ms = val.timestamp_millis();
    let div_factor = match tu {
        TimeUnit::Milliseconds => 1,
        TimeUnit::Microseconds => 1000,
        TimeUnit::Nanoseconds => 1000000,
    };
    return ms * div_factor;
}

fn deserialize_all<'a>(
    bson: &Bson,
    dtype: &DataType,
    ignore_errors: bool,
) -> PolarsResult<AnyValue<'a>> {
    if bson.as_null().is_some() {
        return Ok(AnyValue::Null);
    }
    match dtype {
        DataType::Date => {
            return match bson.as_datetime() {
                Some(val) => Ok(AnyValue::Date(datetime_to_days_since_epoch(val))),
                None => Ok(AnyValue::Null),
            };
        }
        DataType::Datetime(tu, tz) => {
            return Ok(match bson.as_datetime() {
                Some(val) => AnyValue::DatetimeOwned(
                    datetime_to_time_since_epoch(val, *tu),
                    *tu,
                    tz.to_owned().map(|x| Arc::new(x)),
                ),
                None => AnyValue::Null,
            });
        }
        DataType::Float32 => {
            return Ok(match bson.as_f64() {
                Some(val) => AnyValue::Float32(val as f32),
                None => AnyValue::Null,
            });
        }
        DataType::Float64 => {
            return Ok(match bson.as_f64() {
                Some(val) => AnyValue::Float64(val),
                None => AnyValue::Null,
            });
        }
        DataType::String => {
            return Ok(match bson {
                Bson::String(s) => AnyValue::StringOwned(s.into()),
                v => AnyValue::StringOwned(v.to_string().into()),
            });
        }
        dt if dt.is_primitive_numeric() => {
            return Ok(match bson.as_i64() {
                Some(val) => AnyValue::Int64(val),
                None => AnyValue::Null,
            });
        }
        _ => {}
    }
    let out = match bson {
        Bson::ObjectId(obj) => AnyValue::StringOwned(obj.to_hex().into()),
        Bson::Array(arr) => {
            let Some(inner_dtype) = dtype.inner_dtype() else {
                if ignore_errors {
                    return Ok(AnyValue::Null);
                }
                polars_bail!(ComputeError: "expected dtype '{}' in Bson value, got dtype: Array\n\nEncountered value: {}", dtype, bson);
            };
            let vals: Vec<AnyValue> = arr
                .iter()
                .map(|val| deserialize_all(val, inner_dtype, ignore_errors))
                .collect::<PolarsResult<_>>()?;
            let strict = !ignore_errors;
            let s =
                Series::from_any_values_and_dtype(PlSmallStr::EMPTY, &vals, inner_dtype, strict)?;
            AnyValue::List(s)
        }
        Bson::Document(doc) => {
            if let DataType::Struct(fields) = dtype {
                let vals = fields
                    .iter()
                    .map(|field| {
                        if let Some(value) = doc.get(field.name.as_str()) {
                            deserialize_all(value, &field.dtype, ignore_errors)
                        } else {
                            Ok(AnyValue::Null)
                        }
                    })
                    .collect::<PolarsResult<Vec<_>>>()?;
                AnyValue::StructOwned(Box::new((vals, fields.clone())))
            } else {
                if ignore_errors {
                    return Ok(AnyValue::Null);
                }
                polars_bail!(
                    ComputeError: "expected {} in json value, got object", dtype,
                );
            }
        }
        val => AnyValue::StringOwned(format!("{:#?}", val).into()),
    };

    Ok(out)
}

pub fn init_buffers(
    schema: &Schema,
    capacity: usize,
    ignore_errors: bool,
) -> PolarsResult<PlIndexMap<PlSmallStr, DataBuffer>> {
    schema
        .iter()
        .map(|(name, dtype)| {
            let av_buf = (dtype, capacity).into();
            Ok((
                name.to_owned(),
                DataBuffer {
                    name,
                    buf: av_buf,
                    ignore_errors,
                },
            ))
        })
        .collect()
}

pub fn parse_lines(
    docs: Vec<BsonDoc>,
    buffers: &mut PlIndexMap<PlSmallStr, DataBuffer>,
    _ignore_errors: bool,
    _needs_escaping: bool,
    allow_null: bool,
) -> PolarsResult<()> {
    for doc in docs {
        for (s, inner) in buffers.as_mut_slice() {
            match doc.get(s) {
                Some(v) => inner.add(v).expect("unable to parse"),
                None => {
                    if allow_null {
                        inner.add_null();
                    } else {
                        return Err(polars_err!(ComputeError: "received null for key {}", s));
                    }
                }
            }
        }
    }
    Ok(())
}
