# AIRBUS data

The dataset is not redistributed in this repository.

Download the **Airbus Helicopter Accelerometer Dataset** from the ETH Zürich
Research Collection:

- dataset page and permanent identifier: <https://doi.org/10.3929/ethz-b-000415151>
- direct collection page: <https://www.research-collection.ethz.ch/items/a4acb31d-9711-4843-8a60-e2468c0326be>

Download these three files from the collection and place them in `data/raw/`:

```text
data/raw/
├── dftrain.h5
├── dfvalid.h5
└── dfvalid_groundtruth.csv
```

The code expects:

- `dftrain.h5`: 1,677 normal one-minute series, each containing 61,440 samples;
- `dfvalid.h5`: 594 one-minute test series;
- `dfvalid_groundtruth.csv`: one binary label for each test series, with
  `0 = normal` and `1 = anomaly`.

The preparation command validates the filenames, HDF5 structure, dimensions,
ordering, labels, and finite values before creating any cache. The raw files are
opened read-only and are never modified.

Please consult the dataset page for its attribution and reuse conditions.

