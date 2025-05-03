# `polars_bson`: A MongoDB Bson reader for polars

Still a work in progress, will be slowly implementing:

1. Either streaming or SIMD to improve performance
1. Using options for controlling `infer_schema_len`, `n_threads` etc.

> WARNING: The interface for `BsonReader` is still WIP as well. 
> Subject to changes in architecture.

## Sample Usage

```rs
use bson::doc;
use chrono::Utc;

use crate::{BsonReader, common::BsonDoc};

const MONGO_URI: &str = "mongodb://localhost:27017";
const MONGO_DEFAULT_DB: &str = "csdb";

fn main() {
    dotenvy::dotenv().unwrap();
    let modb = match std::env::var("MONGO_LIVEDB") {
        Ok(x) => x,
        Err(_) => {
            return;
        }
    };
    let client = mongodb::sync::Client::with_uri_str(modb).expect("client not built");
    let db = client.database(MONGO_DEFAULT_DB);
    let collection = db.collection::<BsonDoc>("transactions");
    let df = BsonReader::new(collection, doc! {"team.id": 1})
        .finish()
        .unwrap();

    println!("{df:?}");
}
```

## TODO

- [ ] Implement projection
- [ ] Implement schema override
- [ ] Stress test and profiling
- [ ] Test all edge cases
- [ ] Multithreading with chunks or SIMD