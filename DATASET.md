# Dataset Documentation

## Overview

This dataset contains **Spanish household electricity consumption data** collected from smart meters in the Basque region and surrounding provinces. The data spans approximately 5 years (2017-2022) and is organized around the COVID-19 pandemic timeline to enable analysis of behavioral changes in energy consumption patterns.

## Dataset Location

All data files are located in: `data/raw/7362094/`

## File Descriptions

### 1. `metadata.csv`

Household and contract metadata for all users in the dataset.

**Columns:**
- `user`: Anonymized user identifier (SHA-256 hash)
- `start_date`: Beginning of data collection period (ISO 8601 format)
- `end_date`: End of data collection period (ISO 8601 format)
- `length_days`: Duration of monitoring in days
- `length_years`: Duration of monitoring in years
- `potential_samples`: Expected number of hourly samples
- `actual_samples`: Actual number of samples collected
- `missing_samples_abs`: Absolute number of missing samples
- `missing_samples_pct`: Percentage of missing samples
- `contract_start_date`: Start date of electricity contract
- `contract_end_date`: End date of electricity contract (if applicable)
- `contracted_tariff`: Tariff type (e.g., 2.0TD, 2.0DHA, 2.1A, 3.0TD)
- `self_consumption_type`: Self-consumption classification
- `p1` - `p6`: Contracted power limits for different tariff periods (kW)
- `province`: Geographic province (mainly Gipuzkoa, Bizkaia, Araba/Alava, Navarra, Soria, Alicante)
- `municipality`: Municipality/city name
- `zip_code`: Postal code
- `cnae`: National Classification of Economic Activities code

**Geographic Coverage:** Primarily Basque Country (País Vasco) provinces:
- Gipuzkoa
- Bizkaia  
- Araba/Alava
- Navarra
- Some samples from Soria and Alicante

### 2. Time-Series Energy Consumption Files

All time-series files are compressed using Zstandard (`.tzst` format) and contain **tar archives** with multiple CSV files - one CSV file per household. Each household's CSV contains their complete hourly electricity consumption timeline for that period.

**Archive Structure:**
- Each `.tzst` file is a compressed tar archive
- Inside each archive are thousands of individual CSV files
- Each CSV filename is the user's hash (matching the `user` field in `metadata.csv`)
- Each CSV contains hourly readings for one household

**Common CSV Format:**
- `timestamp`: Date and time in format `YYYY-MM-DD HH:MM:SS`
- `kWh`: Energy consumption in kilowatt-hours
- `imputed`: Binary flag (0 = actual reading, 1 = imputed/filled value)

#### `raw.tzst`
Raw, unprocessed energy consumption data as originally collected from smart meters.

**Statistics:**
- **Total households:** 25,559
- **Total hourly readings:** 633,130,317
- **Average readings per household:** ~24,764 (≈1,032 days)

**CSV Format:** One file per user hash  
**Columns:** `timestamp`, `kWh` (no imputation flag)

#### `imp-pre.tzst` - Pre-COVID Period
Energy consumption data **before COVID-19 pandemic**.

**Time Period:** December 2017 - February 2020 (~27 months)  
**Statistics:**
- **Total households:** 12,149
- **Total hourly readings:** 243,796,142
- **Average readings per household:** ~20,067 (≈836 days)

**CSV Format:** One file per user hash  
**Columns:** `timestamp`, `kWh`, `imputed`

**Example:**
```
timestamp,kWh,imputed
2017-12-22 01:00:00,0.17,0
2017-12-22 02:00:00,0.151,0
...
2020-02-29 23:00:00,0.211,0
```

#### `imp-in.tzst` - During-COVID Period
Energy consumption data **during COVID-19 lockdowns and restrictions**.

**Time Period:** March 2020 - May 2021 (~15 months)  
**Statistics:**
- **Total households:** 15,562
- **Total hourly readings:** 169,367,671
- **Average readings per household:** ~10,882 (≈453 days)

**CSV Format:** One file per user hash  
**Columns:** `timestamp`, `kWh`, `imputed`

**Example:**
```
timestamp,kWh,imputed
2020-03-01 00:00:00,0.165,0
2020-03-01 01:00:00,0.168,0
...
2021-05-31 00:00:00,0.291,0
```

#### `imp-post.tzst` - Post-COVID Period
Energy consumption data **after major COVID-19 restrictions ended**.

**Time Period:** May 2021 - June 2022 (~13 months)  
**Statistics:**
- **Total households:** 17,519
- **Total hourly readings:** 155,530,001
- **Average readings per household:** ~8,878 (≈370 days)

**CSV Format:** One file per user hash  
**Columns:** `timestamp`, `kWh`, `imputed`

**Example:**
```
timestamp,kWh,imputed
2021-05-31 00:00:00,0.869,0
2021-05-31 01:00:00,0.157,0
...
2022-06-05 00:00:00,0.297,0
```

## Data Characteristics

### Temporal Resolution
- **Frequency:** Hourly measurements (24 readings per day)
- **Total Duration:** ~5 years (2017-2022)
- **COVID-19 Split:** Three distinct periods for pandemic impact analysis

### Data Quality
- Missing data has been imputed and flagged in the `imputed` column
- Missing sample percentages vary by user (typically <3%)
- Metadata includes completeness metrics for quality assessment

### Dataset Scale Summary

| File | Households | Total Readings | Avg per Household | Time Period |
|------|-----------|----------------|-------------------|-------------|
| `raw.tzst` | 25,559 | 633,130,317 | ~24,764 (≈1,032 days) | Full dataset |
| `imp-pre.tzst` | 12,149 | 243,796,142 | ~20,067 (≈836 days) | Dec 2017 - Feb 2020 |
| `imp-in.tzst` | 15,562 | 169,367,671 | ~10,882 (≈453 days) | Mar 2020 - May 2021 |
| `imp-post.tzst` | 17,519 | 155,530,001 | ~8,878 (≈370 days) | May 2021 - Jun 2022 |

### File Compression
All time-series files use **Zstandard compression** (`.tzst`) for efficient storage.

**To decompress and view:**
```bash
# View first rows across all households
zstd -dc data/raw/7362094/imp-pre.tzst | head -20

# View last rows
zstd -dc data/raw/7362094/imp-post.tzst | tail -20

# Extract full tar archive
zstd -d data/raw/7362094/imp-pre.tzst -o imp-pre.tar

# List all household CSV files in archive
zstd -dc data/raw/7362094/imp-pre.tzst | tar -t | grep '\.csv$'

# Extract a specific household's data
zstd -dc data/raw/7362094/imp-pre.tzst | tar -xO goi4_pre/imp_csv/USER_HASH.csv
```

## Use Cases

This dataset is suitable for:

1. **Temporal Pattern Analysis:** Analyzing how energy consumption patterns changed across pre-COVID, during-COVID, and post-COVID periods
2. **Spatial-Temporal Graph Neural Networks:** Building graphs based on geographic proximity and analyzing consumption patterns
3. **Load Forecasting:** Predicting future energy consumption based on historical patterns
4. **Behavioral Analysis:** Understanding how lockdowns and restrictions affected household energy usage
5. **Anomaly Detection:** Identifying unusual consumption patterns
6. **Continual Learning:** Training models that adapt to distribution shifts across time periods
7. **Energy Efficiency Studies:** Analyzing consumption patterns to identify optimization opportunities

## Citation & Source

Dataset ID: **7362094**

*Note: Add proper citation information when available from the data source.*

## Privacy & Ethics

- All user identifiers are anonymized using SHA-256 hashing
- Location data is limited to province/municipality level
- No personally identifiable information (PII) is included in the dataset
