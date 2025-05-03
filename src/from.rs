use std::borrow::Borrow;

use crate::common::BsonDoc;
use mongodb::bson::Bson;
use polars::frame::row::coerce_dtype;
use polars::prelude::*;

const ITEM_NAME: &str = "item";

#[derive(Debug)]
#[repr(transparent)]
pub struct Wrap<T>(pub T);

impl<T> Clone for Wrap<T>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Wrap(self.0.clone())
    }
}
impl<T> From<T> for Wrap<T> {
    fn from(t: T) -> Self {
        Wrap(t)
    }
}

impl From<&BsonDoc> for Wrap<ArrowDataType> {
    fn from(doc: &BsonDoc) -> Self {
        let fields = doc.iter().map(|(key, value)| {
            let dtype: Wrap<ArrowDataType> = value.into();
            ArrowField::new(key.into(), dtype.0, true)
        });
        ArrowDataType::Struct(fields.collect::<Vec<_>>()).into()
    }
}

impl From<&BsonDoc> for Wrap<DataType> {
    fn from(doc: &BsonDoc) -> Self {
        let fields = doc.iter().map(|(key, value)| {
            let dtype: Wrap<DataType> = value.into();
            Field::new(key.into(), dtype.0)
        });
        DataType::Struct(fields.collect()).into()
    }
}

impl From<&Bson> for Wrap<ArrowDataType> {
    fn from(bson: &Bson) -> Self {
        let dt = match bson {
            Bson::Double(_) => ArrowDataType::Float64,
            Bson::String(_) => ArrowDataType::Utf8,
            Bson::Array(arr) => {
                let dtypes: Vec<_> = arr
                    .iter()
                    .map(|doc| {
                        let dt: Self = doc.into();
                        dt.0
                    })
                    .collect();
                let dtype = if dtypes.is_empty() {
                    ArrowDataType::Null
                } else {
                    coerce_dtype_arrow(&dtypes)
                };
                ArrowDataType::List(Box::new(ArrowField::new("".into(), dtype, true)))
            }
            Bson::Boolean(_) => ArrowDataType::Boolean,
            Bson::Null => ArrowDataType::Null,
            Bson::Int32(_) => ArrowDataType::Int32,
            Bson::Int64(_) => ArrowDataType::Int64,
            Bson::Timestamp(_) => ArrowDataType::Timestamp(ArrowTimeUnit::Second, None),
            Bson::Document(doc) => return doc.into(),
            Bson::DateTime(x) => ArrowDataType::Timestamp(
                ArrowTimeUnit::Millisecond,
                Some(x.to_chrono().timezone().to_string().into()),
            ),
            Bson::ObjectId(_) => ArrowDataType::Utf8,
            Bson::Symbol(_) => ArrowDataType::Utf8,
            Bson::Undefined => ArrowDataType::Unknown,
            _ => ArrowDataType::Utf8,
        };
        Wrap(dt)
    }
}

impl From<&Bson> for Wrap<DataType> {
    fn from(bson: &Bson) -> Self {
        let dt = match bson {
            Bson::Double(_) => DataType::Float64,
            Bson::String(_) => DataType::String,

            Bson::Array(arr) => {
                let dtypes: Vec<_> = arr
                    .iter()
                    .map(|doc| {
                        let dt: Self = doc.into();
                        dt.0
                    })
                    .collect();
                let dtype = if dtypes.is_empty() {
                    DataType::Null
                } else {
                    coerce_dtype(&dtypes)
                };
                DataType::List(Box::new(dtype))
            }
            Bson::Boolean(_) => DataType::Boolean,
            Bson::Null => DataType::Null,
            Bson::Int32(_) => DataType::Int32,
            Bson::Int64(_) => DataType::Int64,
            Bson::Timestamp(_) => DataType::Datetime(TimeUnit::Milliseconds, None),
            Bson::Document(doc) => return doc.into(),
            Bson::DateTime(_) => DataType::Datetime(TimeUnit::Milliseconds, None),
            Bson::ObjectId(_) => DataType::String,
            Bson::Symbol(_) => DataType::String,
            Bson::Undefined => DataType::Unknown(UnknownKind::Any),
            _ => DataType::String,
        };
        Wrap(dt)
    }
}

impl<'a> From<Bson> for Wrap<AnyValue<'a>> {
    fn from(bson: Bson) -> Self {
        let dt = match bson {
            Bson::Double(v) => AnyValue::Float64(v),
            Bson::String(v) => AnyValue::StringOwned(v.into()),
            Bson::Array(arr) => {
                let vals: Vec<Wrap<AnyValue>> = arr.iter().map(|v| v.into()).collect();
                // Wrap is transparent, so this is safe
                let vals =
                    unsafe { std::mem::transmute::<Vec<Wrap<AnyValue>>, Vec<AnyValue>>(vals) };
                let s = Series::new("".into(), vals);
                AnyValue::List(s)
            }
            Bson::Boolean(b) => AnyValue::Boolean(b),
            Bson::Null | Bson::Undefined => AnyValue::Null,
            Bson::Int32(v) => AnyValue::Int32(v),
            Bson::Int64(v) => AnyValue::Int64(v),
            Bson::Timestamp(v) => AnyValue::StringOwned(format!("{v:#?}").into()),
            Bson::DateTime(dt) => {
                AnyValue::Datetime(dt.timestamp_millis(), TimeUnit::Milliseconds, None)
            }
            Bson::Binary(b) => {
                let s = Series::new("".into(), &b.bytes);
                AnyValue::List(s)
            }
            Bson::ObjectId(oid) => AnyValue::StringOwned(oid.to_string().into()),
            Bson::Symbol(s) => AnyValue::StringOwned(s.into()),
            v => AnyValue::StringOwned(format!("{v:#?}").into()),
        };
        Wrap(dt)
    }
}

impl<'a, 'b> From<&'b Bson> for Wrap<AnyValue<'a>> {
    fn from(bson: &'b Bson) -> Self {
        let dt = match bson {
            Bson::Double(v) => AnyValue::Float64(*v),
            Bson::String(v) => AnyValue::StringOwned(v.clone().into()),
            Bson::Array(arr) => {
                let vals: Vec<Wrap<AnyValue>> = arr.iter().map(|v| v.into()).collect();
                // Wrap is transparent, so this is safe
                let vals =
                    unsafe { std::mem::transmute::<Vec<Wrap<AnyValue>>, Vec<AnyValue>>(vals) };
                let s = Series::new("".into(), vals);
                AnyValue::List(s)
            }
            Bson::Boolean(b) => AnyValue::Boolean(*b),
            Bson::Null | Bson::Undefined => AnyValue::Null,
            Bson::Int32(v) => AnyValue::Int32(*v),
            Bson::Int64(v) => AnyValue::Int64(*v),
            Bson::Timestamp(v) => {
                AnyValue::DatetimeOwned((v.time * 1000) as i64, TimeUnit::Milliseconds, None)
            }
            Bson::Binary(b) => {
                let s = Series::new("".into(), &b.bytes);
                AnyValue::List(s)
            }
            Bson::DateTime(dt) => {
                AnyValue::Datetime(dt.timestamp_millis(), TimeUnit::Milliseconds, None)
            }
            Bson::Document(doc) => {
                let vals: (Vec<AnyValue>, Vec<Field>) = doc
                    .into_iter()
                    .map(|(key, value)| {
                        let dt: Wrap<DataType> = value.into();
                        let fld = Field::new(key.into(), dt.0);
                        let av: Wrap<AnyValue<'a>> = value.into();
                        (av.0, fld)
                    })
                    .unzip();

                AnyValue::StructOwned(Box::new(vals))
            }
            Bson::ObjectId(oid) => AnyValue::StringOwned(oid.to_string().into()),
            Bson::Symbol(s) => AnyValue::StringOwned(s.to_string().into()),
            v => AnyValue::StringOwned(format!("{v:#?}").into()),
        };
        Wrap(dt)
    }
}

pub(crate) fn coerce_dtype_arrow<A: Borrow<ArrowDataType>>(datatypes: &[A]) -> ArrowDataType {
    use ArrowDataType::*;

    if datatypes.is_empty() {
        return Null;
    }

    let are_all_equal = datatypes.windows(2).all(|w| w[0].borrow() == w[1].borrow());

    if are_all_equal {
        return datatypes[0].borrow().clone();
    }
    let mut are_all_structs = true;
    let mut are_all_lists = true;
    for dt in datatypes {
        are_all_structs &= matches!(dt.borrow(), Struct(_));
        are_all_lists &= matches!(dt.borrow(), LargeList(_));
    }

    if are_all_structs {
        // all are structs => union of all fields (that may have equal names)
        let fields = datatypes.iter().fold(vec![], |mut acc, dt| {
            if let Struct(new_fields) = dt.borrow() {
                acc.extend(new_fields);
            };
            acc
        });
        // group fields by unique
        let fields = fields.iter().fold(
            PlIndexMap::<&str, PlHashSet<&ArrowDataType>>::default(),
            |mut acc, field| {
                let fieldname = field.name.as_str();
                if !acc.contains_key(fieldname) {
                    acc.insert(fieldname, PlHashSet::default());
                }
                acc.get_mut(fieldname).unwrap().insert(&field.dtype);
                acc
            },
        );
        // and finally, coerce each of the fields within the same name
        let fields = fields
            .into_iter()
            .map(|(name, dts)| {
                let dts = dts.into_iter().collect::<Vec<_>>();
                ArrowField::new(name.into(), coerce_dtype_arrow(&dts), true)
            })
            .collect();
        return Struct(fields);
    } else if are_all_lists {
        let inner_types: Vec<&ArrowDataType> = datatypes
            .iter()
            .map(|dt| {
                if let LargeList(inner) = dt.borrow() {
                    inner.dtype()
                } else {
                    unreachable!();
                }
            })
            .collect();
        return LargeList(Box::new(ArrowField::new(
            PlSmallStr::from_static(ITEM_NAME),
            coerce_dtype_arrow(inner_types.as_slice()),
            true,
        )));
    } else if datatypes.len() > 2 {
        let mut arrow_dtype = ArrowDataType::Null;
        for dtype in datatypes {
            arrow_dtype = coerce_dtype_arrow(&[arrow_dtype, dtype.borrow().to_owned()]);
        }
        return arrow_dtype;
    }
    let (lhs, rhs) = (datatypes[0].borrow(), datatypes[1].borrow());

    match (lhs, rhs) {
        (lhs, rhs) if lhs == rhs => lhs.clone(),
        (LargeList(lhs), LargeList(rhs)) => {
            let inner = coerce_dtype_arrow(&[lhs.dtype(), rhs.dtype()]);
            LargeList(Box::new(ArrowField::new(
                PlSmallStr::from_static(ITEM_NAME),
                inner,
                true,
            )))
        }
        (scalar, LargeList(list)) => {
            let inner = coerce_dtype_arrow(&[scalar, list.dtype()]);
            LargeList(Box::new(ArrowField::new(
                PlSmallStr::from_static(ITEM_NAME),
                inner,
                true,
            )))
        }
        (LargeList(list), scalar) => {
            let inner = coerce_dtype_arrow(&[scalar, list.dtype()]);
            LargeList(Box::new(ArrowField::new(
                PlSmallStr::from_static(ITEM_NAME),
                inner,
                true,
            )))
        }
        (Float64, Int64) => Float64,
        (Int64, Float64) => Float64,
        (Int64, Boolean) => Int64,
        (Boolean, Int64) => Int64,
        (Null, rhs) => rhs.clone(),
        (lhs, Null) => lhs.clone(),
        (_, _) => LargeUtf8,
    }
}
