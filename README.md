
# ID3 Decision Tree Baseline

## Project Overview

This project implements a numerical ID3-style decision tree baseline for classifying hematological diagnosis categories using complete blood count (CBC) measurements.

The baseline uses entropy and information gain to select numerical splitting thresholds.

## Features

The dataset contains the following CBC features:

- WBC
- LYMp
- NEUTp
- LYMn
- NEUTn
- RBC
- HGB
- HCT
- MCV
- MCH
- MCHC
- PLT
- PDW
- PCT

The target column is `Diagnosis`.

## Requirements

- Python
- pandas
- scikit-learn

Install the dependencies:

```bash
pip install pandas scikit-learn
```

## Running the Baseline

1. Obtain the dataset from its source and verify its license.
2. Place the permitted CSV file in the project directory.
3. Run:

```bash
python id3_numeric_baseline.py
```

4. Enter the CSV filename when prompted.
5. Enter `Diagnosis` as the target column.

## Preliminary Results

A preliminary evaluation on a screened dataset containing 1,212 records used 909 training records and 303 testing records, producing 99.01% accuracy.

These results have not yet been fully validated. Dataset provenance, data quality, potential label leakage, and evaluation methodology require further investigation.

## Disclaimer

This is an academic research prototype, not a clinically validated diagnostic system. Predictions must not be used for medical diagnosis or treatment decisions.
