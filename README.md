# phenoatlas

**Python port of [phenofit](https://github.com/eco-hydro/phenofit) — vegetation phenology extraction from remote sensing time-series.**

> Work in progress. API mirrors the R package phenofit (Kong et al., 2022).

## Features

- **Smoothing**: Weighted Whittaker, Savitzky-Golay, HANTS
- **Curve fitting**: 6 double-logistic methods (AG, Beck, Elmore, Gu, Klos, Zhang)
- **Season detection**: Automatic growing season identification
- **Phenology extraction**: TRS, DER, Gu, Zhang methods
- **QC decoders**: MODIS, Sentinel-2, SPOT, GIMMS NDVI3g

## Installation

```bash
pip install phenoatlas
```

## Quick Start

```python
import phenoatlas as pa

# Smooth a time series
y_smooth = pa.smooth_whit(y, w=weights, nptperyear=23)

# Detect growing seasons
seasons = pa.season_mov(y, t, nptperyear=23)

# Fit curves to one season
fit = pa.curvefit(y_season, t_season, methods=["Beck", "Elmore"])

# Extract phenological dates
pheno = pa.get_pheno(fit)
```

## References

Kong, D., et al. (2022). phenofit: An R package for extracting vegetation phenology
from time series remote sensing data. *Methods in Ecology and Evolution*.
