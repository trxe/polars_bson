use crate::common::BsonDoc;
use mongodb::bson::Bson;
use polars::frame::row::coerce_dtype;
use polars::prelude::*;

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

impl From<&BsonDoc> for Wrap<DataType> {
    fn from(doc: &BsonDoc) -> Self {
        let fields = doc.iter().map(|(key, value)| {
            let dtype: Wrap<DataType> = value.into();
            Field::new(key.into(), dtype.0)
        });
        DataType::Struct(fields.collect()).into()
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
