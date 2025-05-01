use std::mem::MaybeUninit;

use polars::error::to_compute_err;
use polars::prelude::CsvEncoding;
use polars_core::prelude::*;
use polars_core::utils::arrow::array::MutableBinaryViewArray;
use polars_core::utils::arrow::legacy::trusted_len::TrustedLenPush;
use polars_time::chunkedarray::string::Pattern;
use polars_time::prelude::string::infer::{
    DatetimeInfer, StrpTimeParser, TryFromWithUnit, infer_pattern_single,
};

use crate::common::{SyncCursor, is_whitespace, skip_whitespace};

pub fn parse_lines<'a>(
    mut cursor: SyncCursor,
    buffers: &mut PlIndexMap<PlSmallStr, Buffer>,
    ignore_errors: bool,
    needs_escaping: bool,
    allow_null: bool,
) -> mongodb::error::Result<()> {
    while let Some(Ok(doc)) = cursor.next() {
        buffers.iter_mut().for_each(|(s, inner)| match doc.get(s) {
            Some(v) => {
                let bts = v.to_string().bytes().collect::<Vec<u8>>();
                inner
                    .add(bts.as_slice(), ignore_errors, needs_escaping, !allow_null)
                    .expect("unable to parse")
            }
            None => inner.add_null(allow_null),
        });
    }
    Ok(())
}

fn escape_field(bytes: &[u8], quote: u8, buf: &mut [MaybeUninit<u8>]) -> usize {
    debug_assert!(bytes.len() > 1);
    let mut prev_quote = false;

    let mut count = 0;
    for c in unsafe { bytes.get_unchecked(1..bytes.len() - 1) } {
        if *c == quote {
            if prev_quote {
                prev_quote = false;
                unsafe { buf.get_unchecked_mut(count).write(*c) };
                count += 1;
            } else {
                prev_quote = true;
            }
        } else {
            prev_quote = false;
            unsafe { buf.get_unchecked_mut(count).write(*c) };
            count += 1;
        }
    }
    count
}
pub(crate) trait PrimitiveParser: PolarsNumericType {
    fn parse(bytes: &[u8]) -> Option<Self::Native>;
}

impl PrimitiveParser for Float32Type {
    #[inline]
    fn parse(bytes: &[u8]) -> Option<f32> {
        fast_float2::parse(bytes).ok()
    }
}
impl PrimitiveParser for Float64Type {
    #[inline]
    fn parse(bytes: &[u8]) -> Option<f64> {
        fast_float2::parse(bytes).ok()
    }
}

impl PrimitiveParser for UInt8Type {
    #[inline]
    fn parse(bytes: &[u8]) -> Option<u8> {
        atoi_simd::parse_skipped(bytes).ok()
    }
}
impl PrimitiveParser for UInt16Type {
    #[inline]
    fn parse(bytes: &[u8]) -> Option<u16> {
        atoi_simd::parse_skipped(bytes).ok()
    }
}
impl PrimitiveParser for UInt32Type {
    #[inline]
    fn parse(bytes: &[u8]) -> Option<u32> {
        atoi_simd::parse_skipped(bytes).ok()
    }
}
impl PrimitiveParser for UInt64Type {
    #[inline]
    fn parse(bytes: &[u8]) -> Option<u64> {
        atoi_simd::parse_skipped(bytes).ok()
    }
}
impl PrimitiveParser for Int8Type {
    #[inline]
    fn parse(bytes: &[u8]) -> Option<i8> {
        atoi_simd::parse_skipped(bytes).ok()
    }
}
impl PrimitiveParser for Int16Type {
    #[inline]
    fn parse(bytes: &[u8]) -> Option<i16> {
        atoi_simd::parse_skipped(bytes).ok()
    }
}
impl PrimitiveParser for Int32Type {
    #[inline]
    fn parse(bytes: &[u8]) -> Option<i32> {
        atoi_simd::parse_skipped(bytes).ok()
    }
}
impl PrimitiveParser for Int64Type {
    #[inline]
    fn parse(bytes: &[u8]) -> Option<i64> {
        atoi_simd::parse_skipped(bytes).ok()
    }
}
impl PrimitiveParser for Int128Type {
    #[inline]
    fn parse(bytes: &[u8]) -> Option<i128> {
        atoi_simd::parse_skipped(bytes).ok()
    }
}

trait ParsedBuffer {
    fn parse_bytes(
        &mut self,
        bytes: &[u8],
        ignore_errors: bool,
        _needs_escaping: bool,
        _missing_is_null: bool,
        _time_unit: Option<TimeUnit>,
    ) -> PolarsResult<()>;
}

impl<T> ParsedBuffer for PrimitiveChunkedBuilder<T>
where
    T: PolarsNumericType + PrimitiveParser,
{
    #[inline]
    fn parse_bytes(
        &mut self,
        bytes: &[u8],
        ignore_errors: bool,
        needs_escaping: bool,
        _missing_is_null: bool,
        _time_unit: Option<TimeUnit>,
    ) -> PolarsResult<()> {
        if bytes.is_empty() {
            self.append_null()
        } else {
            let bytes = if needs_escaping {
                &bytes[1..bytes.len() - 1]
            } else {
                bytes
            };

            // legacy comment (remember this if you decide to use Results again):
            // its faster to work on options.
            // if we need to throw an error, we parse again to be able to throw the error

            match T::parse(bytes) {
                Some(value) => self.append_value(value),
                None => {
                    // try again without whitespace
                    if !bytes.is_empty() && is_whitespace(bytes[0]) {
                        let bytes = skip_whitespace(bytes);
                        return self.parse_bytes(
                            bytes,
                            ignore_errors,
                            false, // escaping was already done
                            _missing_is_null,
                            None,
                        );
                    }
                    polars_ensure!(
                        bytes.is_empty() || ignore_errors,
                        ComputeError: "remaining bytes non-empty",
                    );
                    self.append_null()
                }
            };
        }
        Ok(())
    }
}

pub struct Utf8Field {
    name: PlSmallStr,
    mutable: MutableBinaryViewArray<[u8]>,
    scratch: Vec<u8>,
    quote_char: u8,
    encoding: CsvEncoding,
}

impl Utf8Field {
    fn new(
        name: PlSmallStr,
        capacity: usize,
        quote_char: Option<u8>,
        encoding: CsvEncoding,
    ) -> Self {
        Self {
            name,
            mutable: MutableBinaryViewArray::with_capacity(capacity),
            scratch: vec![],
            quote_char: quote_char.unwrap_or(b'"'),
            encoding,
        }
    }
}

#[inline]
pub fn validate_utf8(bytes: &[u8]) -> bool {
    simdutf8::basic::from_utf8(bytes).is_ok()
}

impl ParsedBuffer for Utf8Field {
    #[inline]
    fn parse_bytes(
        &mut self,
        bytes: &[u8],
        ignore_errors: bool,
        needs_escaping: bool,
        missing_is_null: bool,
        _time_unit: Option<TimeUnit>,
    ) -> PolarsResult<()> {
        if bytes.is_empty() {
            if missing_is_null {
                self.mutable.push_null()
            } else {
                self.mutable.push(Some([]))
            }
            return Ok(());
        }

        // note that one branch writes without updating the length, so we must do that later.
        let escaped_bytes = if needs_escaping {
            self.scratch.clear();
            self.scratch.reserve(bytes.len());
            polars_ensure!(bytes.len() > 1 && bytes.last() == Some(&self.quote_char), ComputeError: "invalid csv file\n\nField `{}` is not properly escaped.", std::str::from_utf8(bytes).map_err(to_compute_err)?);

            // SAFETY:
            // we just allocated enough capacity and data_len is correct.
            unsafe {
                let n_written =
                    escape_field(bytes, self.quote_char, self.scratch.spare_capacity_mut());
                self.scratch.set_len(n_written);
            }

            self.scratch.as_slice()
        } else {
            bytes
        };

        if matches!(self.encoding, CsvEncoding::LossyUtf8) | ignore_errors {
            // It is important that this happens after escaping, as invalid escaped string can produce
            // invalid utf8.
            let parse_result = validate_utf8(escaped_bytes);

            match parse_result {
                true => {
                    let value = escaped_bytes;
                    self.mutable.push_value(value)
                }
                false => {
                    if matches!(self.encoding, CsvEncoding::LossyUtf8) {
                        // TODO! do this without allocating
                        let s = String::from_utf8_lossy(escaped_bytes);
                        self.mutable.push_value(s.as_ref().as_bytes())
                    } else if ignore_errors {
                        self.mutable.push_null()
                    } else {
                        // If field before escaping is valid utf8, the escaping is incorrect.
                        if needs_escaping && validate_utf8(bytes) {
                            polars_bail!(ComputeError: "string field is not properly escaped");
                        } else {
                            polars_bail!(ComputeError: "invalid utf-8 sequence");
                        }
                    }
                }
            }
        } else {
            self.mutable.push_value(escaped_bytes)
        }

        Ok(())
    }
}

pub struct CategoricalField {
    escape_scratch: Vec<u8>,
    quote_char: u8,
    builder: CategoricalChunkedBuilder,
    is_enum: bool,
}

impl CategoricalField {
    fn new(
        name: PlSmallStr,
        capacity: usize,
        quote_char: Option<u8>,
        ordering: CategoricalOrdering,
    ) -> Self {
        let builder = CategoricalChunkedBuilder::new(name, capacity, ordering);

        Self {
            escape_scratch: vec![],
            quote_char: quote_char.unwrap_or(b'"'),
            builder,
            is_enum: false,
        }
    }

    fn new_enum(quote_char: Option<u8>, builder: CategoricalChunkedBuilder) -> Self {
        Self {
            escape_scratch: vec![],
            quote_char: quote_char.unwrap_or(b'"'),
            builder,
            is_enum: true,
        }
    }

    #[inline]
    fn parse_bytes(
        &mut self,
        bytes: &[u8],
        ignore_errors: bool,
        needs_escaping: bool,
        _missing_is_null: bool,
        _time_unit: Option<TimeUnit>,
    ) -> PolarsResult<()> {
        if bytes.is_empty() {
            self.builder.append_null();
            return Ok(());
        }
        if validate_utf8(bytes) {
            if needs_escaping {
                polars_ensure!(bytes.len() > 1, ComputeError: "invalid csv file\n\nField `{}` is not properly escaped.", std::str::from_utf8(bytes).map_err(to_compute_err)?);
                self.escape_scratch.clear();
                self.escape_scratch.reserve(bytes.len());
                // SAFETY:
                // we just allocated enough capacity and data_len is correct.
                unsafe {
                    let n_written = escape_field(
                        bytes,
                        self.quote_char,
                        self.escape_scratch.spare_capacity_mut(),
                    );
                    self.escape_scratch.set_len(n_written);
                }

                // SAFETY:
                // just did utf8 check
                let key = unsafe { std::str::from_utf8_unchecked(&self.escape_scratch) };
                if self.is_enum {
                    self.builder.append_value(key);
                } else {
                    self.builder.append_value(key);
                }
            } else {
                // SAFETY:
                // just did utf8 check
                let key = unsafe { std::str::from_utf8_unchecked(bytes) };
                if self.is_enum {
                    self.builder.append_value(key)
                } else {
                    self.builder.append_value(key)
                }
            }
        } else if ignore_errors {
            self.builder.append_null()
        } else {
            polars_bail!(ComputeError: "invalid utf-8 sequence");
        }
        Ok(())
    }
}

impl ParsedBuffer for BooleanChunkedBuilder {
    #[inline]
    fn parse_bytes(
        &mut self,
        bytes: &[u8],
        ignore_errors: bool,
        needs_escaping: bool,
        _missing_is_null: bool,
        _time_unit: Option<TimeUnit>,
    ) -> PolarsResult<()> {
        let bytes = if needs_escaping {
            &bytes[1..bytes.len() - 1]
        } else {
            bytes
        };
        if bytes.eq_ignore_ascii_case(b"false") {
            self.append_value(false);
        } else if bytes.eq_ignore_ascii_case(b"true") {
            self.append_value(true);
        } else if ignore_errors || bytes.is_empty() {
            self.append_null();
        } else {
            polars_bail!(
                ComputeError: "error while parsing value {} as boolean",
                String::from_utf8_lossy(bytes),
            );
        }
        Ok(())
    }
}

pub struct DatetimeField<T: PolarsNumericType> {
    compiled: Option<DatetimeInfer<T>>,
    builder: PrimitiveChunkedBuilder<T>,
}

impl<T: PolarsNumericType> DatetimeField<T> {
    fn new(name: PlSmallStr, capacity: usize) -> Self {
        let builder = PrimitiveChunkedBuilder::<T>::new(name, capacity);
        Self {
            compiled: None,
            builder,
        }
    }
}

fn slow_datetime_parser<T>(
    buf: &mut DatetimeField<T>,
    bytes: &[u8],
    time_unit: Option<TimeUnit>,
    ignore_errors: bool,
) -> PolarsResult<()>
where
    T: PolarsNumericType,
    DatetimeInfer<T>: TryFromWithUnit<Pattern>,
{
    let val = if bytes.is_ascii() {
        // SAFETY:
        // we just checked it is ascii
        unsafe { std::str::from_utf8_unchecked(bytes) }
    } else {
        match std::str::from_utf8(bytes) {
            Ok(val) => val,
            Err(_) => {
                if ignore_errors {
                    buf.builder.append_null();
                    return Ok(());
                } else {
                    polars_bail!(ComputeError: "invalid utf-8 sequence");
                }
            }
        }
    };

    let pattern = match &buf.compiled {
        Some(compiled) => compiled.pattern,
        None => match infer_pattern_single(val) {
            Some(pattern) => pattern,
            None => {
                if ignore_errors {
                    buf.builder.append_null();
                    return Ok(());
                } else {
                    polars_bail!(ComputeError: "could not find a 'date/datetime' pattern for '{}'", val)
                }
            }
        },
    };
    match DatetimeInfer::try_from_with_unit(pattern, time_unit) {
        Ok(mut infer) => {
            let parsed = infer.parse(val);
            let Some(parsed) = parsed else {
                if ignore_errors {
                    buf.builder.append_null();
                    return Ok(());
                } else {
                    polars_bail!(ComputeError: "could not parse '{}' with pattern '{:?}'", val, pattern)
                }
            };

            buf.compiled = Some(infer);
            buf.builder.append_value(parsed);
            Ok(())
        }
        Err(err) => {
            if ignore_errors {
                buf.builder.append_null();
                Ok(())
            } else {
                Err(err)
            }
        }
    }
}

impl<T> ParsedBuffer for DatetimeField<T>
where
    T: PolarsNumericType,
    DatetimeInfer<T>: TryFromWithUnit<Pattern> + StrpTimeParser<T::Native>,
{
    #[inline]
    fn parse_bytes(
        &mut self,
        mut bytes: &[u8],
        ignore_errors: bool,
        needs_escaping: bool,
        _missing_is_null: bool,
        time_unit: Option<TimeUnit>,
    ) -> PolarsResult<()> {
        if needs_escaping && bytes.len() >= 2 {
            bytes = &bytes[1..bytes.len() - 1]
        }

        if bytes.is_empty() {
            // for types other than string `_missing_is_null` is irrelevant; we always append null
            self.builder.append_null();
            return Ok(());
        }

        match &mut self.compiled {
            None => slow_datetime_parser(self, bytes, time_unit, ignore_errors),
            Some(compiled) => {
                match compiled.parse_bytes(bytes, time_unit) {
                    Some(parsed) => {
                        self.builder.append_value(parsed);
                        Ok(())
                    }
                    // fall back on chrono parser
                    // this is a lot slower, we need to do utf8 checking and use
                    // the slower parser
                    None => slow_datetime_parser(self, bytes, time_unit, ignore_errors),
                }
            }
        }
    }
}

pub fn init_buffers(
    capacity: usize,
    schema: &Schema,
    quote_char: Option<u8>,
    encoding: CsvEncoding,
    decimal_comma: bool,
) -> PolarsResult<PlIndexMap<PlSmallStr, Buffer>> {
    schema
        .iter()
        .map(|(colname, dtype)| {
            let name = colname.to_owned();
            let builder = match dtype {
                DataType::Boolean => Buffer::Boolean(BooleanChunkedBuilder::new(name, capacity)),
                DataType::Int8 => Buffer::Int8(PrimitiveChunkedBuilder::new(name, capacity)),
                DataType::Int16 => Buffer::Int16(PrimitiveChunkedBuilder::new(name, capacity)),
                DataType::Int32 => Buffer::Int32(PrimitiveChunkedBuilder::new(name, capacity)),
                DataType::Int64 => Buffer::Int64(PrimitiveChunkedBuilder::new(name, capacity)),
                DataType::Int128 => Buffer::Int128(PrimitiveChunkedBuilder::new(name, capacity)),
                DataType::UInt8 => Buffer::UInt8(PrimitiveChunkedBuilder::new(name, capacity)),
                DataType::UInt16 => Buffer::UInt16(PrimitiveChunkedBuilder::new(name, capacity)),
                DataType::UInt32 => Buffer::UInt32(PrimitiveChunkedBuilder::new(name, capacity)),
                DataType::UInt64 => Buffer::UInt64(PrimitiveChunkedBuilder::new(name, capacity)),
                DataType::Float32 => {
                    if decimal_comma {
                        Buffer::DecimalFloat32(
                            PrimitiveChunkedBuilder::new(name, capacity),
                            Default::default(),
                        )
                    } else {
                        Buffer::Float32(PrimitiveChunkedBuilder::new(name, capacity))
                    }
                }
                DataType::Float64 => {
                    if decimal_comma {
                        Buffer::DecimalFloat64(
                            PrimitiveChunkedBuilder::new(name, capacity),
                            Default::default(),
                        )
                    } else {
                        Buffer::Float64(PrimitiveChunkedBuilder::new(name, capacity))
                    }
                }
                DataType::String => {
                    Buffer::Utf8(Utf8Field::new(name, capacity, quote_char, encoding))
                }
                DataType::Datetime(time_unit, time_zone) => Buffer::Datetime {
                    buf: DatetimeField::new(name, capacity),
                    time_unit: *time_unit,
                    time_zone: time_zone.clone(),
                },
                DataType::Date => Buffer::Date(DatetimeField::new(name, capacity)),
                DataType::Categorical(_, ordering) => Buffer::Categorical(CategoricalField::new(
                    name, capacity, quote_char, *ordering,
                )),
                DataType::Enum(rev_map, _) => {
                    let Some(rev_map) = rev_map else {
                        polars_bail!(ComputeError: "enum categories must be set")
                    };
                    let cats = rev_map.get_categories();
                    let mut builder =
                        CategoricalChunkedBuilder::new(name, capacity, Default::default());
                    for cat in cats.values_iter() {
                        builder.register_value(cat);
                    }
                    Buffer::Categorical(CategoricalField::new_enum(quote_char, builder))
                }
                dt => polars_bail!(
                    ComputeError: "unsupported data type when reading CSV: {} when reading CSV", dt,
                ),
            };
            Ok((PlSmallStr::from_str(&colname), builder))
        })
        .collect()
}

#[allow(clippy::large_enum_variant)]
pub enum Buffer {
    Boolean(BooleanChunkedBuilder),
    Int8(PrimitiveChunkedBuilder<Int8Type>),
    Int16(PrimitiveChunkedBuilder<Int16Type>),
    Int32(PrimitiveChunkedBuilder<Int32Type>),
    Int64(PrimitiveChunkedBuilder<Int64Type>),
    Int128(PrimitiveChunkedBuilder<Int128Type>),
    UInt8(PrimitiveChunkedBuilder<UInt8Type>),
    UInt16(PrimitiveChunkedBuilder<UInt16Type>),
    UInt32(PrimitiveChunkedBuilder<UInt32Type>),
    UInt64(PrimitiveChunkedBuilder<UInt64Type>),
    Float32(PrimitiveChunkedBuilder<Float32Type>),
    Float64(PrimitiveChunkedBuilder<Float64Type>),
    /// Stores the Utf8 fields and the total string length seen for that column
    Utf8(Utf8Field),
    Datetime {
        buf: DatetimeField<Int64Type>,
        time_unit: TimeUnit,
        time_zone: Option<TimeZone>,
    },
    Date(DatetimeField<Int32Type>),
    #[allow(dead_code)]
    Categorical(CategoricalField),
    DecimalFloat32(PrimitiveChunkedBuilder<Float32Type>, Vec<u8>),
    DecimalFloat64(PrimitiveChunkedBuilder<Float64Type>, Vec<u8>),
}

impl Buffer {
    pub fn into_series(self) -> PolarsResult<Series> {
        let s = match self {
            Buffer::Boolean(v) => v.finish().into_series(),
            Buffer::Int8(v) => v.finish().into_series(),
            Buffer::Int16(v) => v.finish().into_series(),
            Buffer::Int32(v) => v.finish().into_series(),
            Buffer::Int64(v) => v.finish().into_series(),
            Buffer::Int128(v) => v.finish().into_series(),
            Buffer::UInt8(v) => v.finish().into_series(),
            Buffer::UInt16(v) => v.finish().into_series(),
            Buffer::UInt32(v) => v.finish().into_series(),
            Buffer::UInt64(v) => v.finish().into_series(),
            Buffer::Float32(v) => v.finish().into_series(),
            Buffer::Float64(v) => v.finish().into_series(),
            Buffer::DecimalFloat32(v, _) => v.finish().into_series(),
            Buffer::DecimalFloat64(v, _) => v.finish().into_series(),
            Buffer::Datetime {
                buf,
                time_unit,
                time_zone,
            } => buf
                .builder
                .finish()
                .into_series()
                .cast(&DataType::Datetime(time_unit, time_zone))
                .unwrap(),
            Buffer::Date(v) => v
                .builder
                .finish()
                .into_series()
                .cast(&DataType::Date)
                .unwrap(),

            Buffer::Utf8(v) => {
                let arr = v.mutable.freeze();
                StringChunked::with_chunk(v.name.clone(), unsafe { arr.to_utf8view_unchecked() })
                    .into_series()
            }
            #[allow(unused_variables)]
            Buffer::Categorical(buf) => {
                let ca = buf.builder.finish();

                if buf.is_enum {
                    let DataType::Categorical(Some(rev_map), _) = ca.dtype() else {
                        unreachable!()
                    };
                    let idx = ca.physical().clone();
                    let dtype = DataType::Enum(Some(rev_map.clone()), Default::default());

                    unsafe {
                        CategoricalChunked::from_cats_and_dtype_unchecked(idx, dtype).into_series()
                    }
                } else {
                    ca.into_series()
                }
            }
        };
        Ok(s)
    }

    pub fn add_null(&mut self, valid: bool) {
        match self {
            Buffer::Boolean(v) => v.append_null(),
            Buffer::Int8(v) => v.append_null(),
            Buffer::Int16(v) => v.append_null(),
            Buffer::Int32(v) => v.append_null(),
            Buffer::Int64(v) => v.append_null(),
            Buffer::Int128(v) => v.append_null(),
            Buffer::UInt8(v) => v.append_null(),
            Buffer::UInt16(v) => v.append_null(),
            Buffer::UInt32(v) => v.append_null(),
            Buffer::UInt64(v) => v.append_null(),
            Buffer::Float32(v) => v.append_null(),
            Buffer::Float64(v) => v.append_null(),
            Buffer::DecimalFloat32(v, _) => v.append_null(),
            Buffer::DecimalFloat64(v, _) => v.append_null(),
            Buffer::Utf8(v) => {
                if valid {
                    v.mutable.push_value("")
                } else {
                    v.mutable.push_null()
                }
            }
            Buffer::Datetime { buf, .. } => buf.builder.append_null(),
            Buffer::Date(v) => v.builder.append_null(),
            Buffer::Categorical(cat_builder) => cat_builder.builder.append_null(),
        };
    }

    pub fn dtype(&self) -> DataType {
        match self {
            Buffer::Boolean(_) => DataType::Boolean,
            Buffer::Int8(_) => DataType::Int8,
            Buffer::Int16(_) => DataType::Int16,
            Buffer::Int32(_) => DataType::Int32,
            Buffer::Int64(_) => DataType::Int64,
            Buffer::Int128(_) => DataType::Int128,
            Buffer::UInt8(_) => DataType::UInt8,
            Buffer::UInt16(_) => DataType::UInt16,
            Buffer::UInt32(_) => DataType::UInt32,
            Buffer::UInt64(_) => DataType::UInt64,
            Buffer::Float32(_) | Buffer::DecimalFloat32(_, _) => DataType::Float32,
            Buffer::Float64(_) | Buffer::DecimalFloat64(_, _) => DataType::Float64,
            Buffer::Utf8(_) => DataType::String,
            Buffer::Datetime { time_unit, .. } => DataType::Datetime(*time_unit, None),
            Buffer::Date(_) => DataType::Date,
            Buffer::Categorical(_) => DataType::Categorical(None, Default::default()),
        }
    }

    #[inline]
    pub fn add(
        &mut self,
        bytes: &[u8],
        ignore_errors: bool,
        needs_escaping: bool,
        missing_is_null: bool,
    ) -> PolarsResult<()> {
        use Buffer::*;
        match self {
            Boolean(buf) => <BooleanChunkedBuilder as ParsedBuffer>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
                None,
            ),
            Int8(buf) => <PrimitiveChunkedBuilder<Int8Type> as ParsedBuffer>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
                None,
            ),
            Int16(buf) => <PrimitiveChunkedBuilder<Int16Type> as ParsedBuffer>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
                None,
            ),
            Int32(buf) => <PrimitiveChunkedBuilder<Int32Type> as ParsedBuffer>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
                None,
            ),
            Int64(buf) => <PrimitiveChunkedBuilder<Int64Type> as ParsedBuffer>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
                None,
            ),
            Int128(buf) => <PrimitiveChunkedBuilder<Int128Type> as ParsedBuffer>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
                None,
            ),
            UInt8(buf) => <PrimitiveChunkedBuilder<UInt8Type> as ParsedBuffer>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
                None,
            ),
            UInt16(buf) => <PrimitiveChunkedBuilder<UInt16Type> as ParsedBuffer>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
                None,
            ),
            UInt32(buf) => <PrimitiveChunkedBuilder<UInt32Type> as ParsedBuffer>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
                None,
            ),
            UInt64(buf) => <PrimitiveChunkedBuilder<UInt64Type> as ParsedBuffer>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
                None,
            ),
            Float32(buf) => <PrimitiveChunkedBuilder<Float32Type> as ParsedBuffer>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
                None,
            ),
            Float64(buf) => <PrimitiveChunkedBuilder<Float64Type> as ParsedBuffer>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
                None,
            ),
            DecimalFloat32(buf, scratch) => {
                prepare_decimal_comma(bytes, scratch);
                <PrimitiveChunkedBuilder<Float32Type> as ParsedBuffer>::parse_bytes(
                    buf,
                    scratch,
                    ignore_errors,
                    needs_escaping,
                    missing_is_null,
                    None,
                )
            }
            DecimalFloat64(buf, scratch) => {
                prepare_decimal_comma(bytes, scratch);
                <PrimitiveChunkedBuilder<Float64Type> as ParsedBuffer>::parse_bytes(
                    buf,
                    scratch,
                    ignore_errors,
                    needs_escaping,
                    missing_is_null,
                    None,
                )
            }
            Utf8(buf) => <Utf8Field as ParsedBuffer>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
                None,
            ),
            Datetime { buf, time_unit, .. } => {
                <DatetimeField<Int64Type> as ParsedBuffer>::parse_bytes(
                    buf,
                    bytes,
                    ignore_errors,
                    needs_escaping,
                    missing_is_null,
                    Some(*time_unit),
                )
            }
            Date(buf) => <DatetimeField<Int32Type> as ParsedBuffer>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                needs_escaping,
                missing_is_null,
                None,
            ),
            Categorical(buf) => {
                buf.parse_bytes(bytes, ignore_errors, needs_escaping, missing_is_null, None)
            }
        }
    }
}

#[inline]
fn prepare_decimal_comma(bytes: &[u8], scratch: &mut Vec<u8>) {
    scratch.clear();
    scratch.reserve(bytes.len());

    // SAFETY: we pre-allocated.
    for &byte in bytes {
        if byte == b',' {
            unsafe { scratch.push_unchecked(b'.') }
        } else {
            unsafe { scratch.push_unchecked(byte) }
        }
    }
}
