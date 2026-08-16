# Dataset placement

No dataset is distributed with this repository.

## Criteo Uplift Prediction Dataset

Obtain Criteo Uplift v2.1 from the [Criteo AI Lab source page](https://ailab.criteo.com/criteo-uplift-prediction-dataset/) and follow its stated terms of use. The source page identifies the dataset license as Creative Commons Attribution-NonCommercial-ShareAlike 4.0.

Place the downloaded file at:

```text
data/criteo-uplift/criteo-uplift-v2.1.csv.gz
```

The Criteo-MT7 semi-synthetic generator reads only user-supplied Criteo features and generates treatments and outcomes locally. Generated rows are not included in the repository.

## Hillstrom MineThatData challenge

The optional loader documents the expected filenames in `code/data_gen/hillstrom_loader.py`. Obtain the data directly from the [original MineThatData source](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html) and verify the applicable terms before use. The dataset is not redistributed here, and the E1 driver is not included in v1.0.0.

## Restricted data

Never place restricted industrial data, user-level outputs, credentials, or private path markers inside a public clone. Industrial loaders and schemas are deliberately excluded from this release.
