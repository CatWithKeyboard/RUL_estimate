# Bearing RUL Application

This project estimates the remaining useful life (RUL) of bearings using the IMS/NASA run-to-failure dataset.

## Files
- `data_loader.py` - loads IMS dataset files and extracts bearing channels
- `feature_engineering.py` - time-domain and frequency-domain features
- `labeling.py` - creates RUL labels
- `train.py` - builds feature table and trains the regression model
- `predict.py` - batch prediction for a folder of files
- `app.py` - Streamlit web app

## Expected dataset structure

```text
ims-data/
  set1/
    2003.10.22.12.06.24
    ...
  set2/
    2004.02.12.10.32.39
    ...
  set3/
    2004.03.04.09.27.46
    ...
```

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py --data-root /path/to/ims-data --output-dir artifacts
```

## Predict from a folder

```bash
python predict.py --model-path artifacts/bearing_rul_model.joblib --dataset set2 --bearing-id 1 --folder-path /path/to/new/files --output-csv predictions.csv
```

## Run Streamlit app

```bash
streamlit run app.py
```

## Notes
- The current baseline trains on failed bearings only.
- Set 1 combines the two channels belonging to the same bearing by averaging them.
- The model is a Random Forest baseline and can be upgraded later to XGBoost or LSTM.
