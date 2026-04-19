# Deep Learning

This repo contains my deep learning assignments. Each folder is a separate problem with its own dataset, preprocessing, and model. I built each one in both Keras and PyTorch where it made sense, and I kept a preprocessing notebook separate from the modeling notebook.

---

## classification

**Dataset:** Stellar Classification from SDSS DR17 (Kaggle). 100k observations, three classes: Star, Galaxy, QSO.

**What the data is:** Sky objects measured by the Sloan Digital Sky Survey. Features include photometric filter magnitudes (u, g, r, i, z) across different wavelengths and redshift. I dropped all the metadata columns (IDs, plate, fiber, etc.) since they carry no signal.

**Feature engineering:** I created color indices by subtracting adjacent filter bands (u-g, g-r, r-i, i-z). In astronomy these differences remove distance dependence and are the standard way to separate object types.

**Model:** Dense feedforward network (64 - 32 - 16 - softmax). Classes are imbalanced so I used `class_weight` instead of undersampling.

**Notebooks:** `Pre_Pro.ipynb` for cleaning and feature engineering, `keras.ipynb` and `pytorch.ipynb` for training and evaluation.

---

## Regression

**Dataset:** Steel industry energy consumption from a South Korean steel company. One year of data recorded every 15 minutes (35,040 rows). Target: `Usage_kWh`.

**What the data is:** Power measurements including reactive power (lagging and leading), power factor, CO2 concentration, time of day (seconds from midnight), weekday/weekend status, and load type.

**Approach:** I used permutation importance from a Random Forest to drop low-importance features before training. The split is chronological (70/15/15) to respect the time structure.

**Models:** Keras with `keras_tuner` for hyperparameter search, PyTorch, and XGBoost.

**Notebooks:** `Explor_Prepros.ipynb` for exploration and preprocessing, `Keras.ipynb`, `PyTorch.ipynb`, and `Xgboot.ipynb` for the three model implementations.

---

## LSTM

**Dataset:** Power consumption of Tetouan City, Morocco (UCI). About one year of data at 10-minute intervals, three target zones.

**What the data is:** Multivariate time series with weather features (temperature, humidity, wind speed, diffuse flows) and three power consumption zones as targets.

**Preprocessing:** I resampled to hourly, applied cyclical encoding for hour and month (sin/cos), and split chronologically.

**Model:** LSTM with a Conv1D layer upfront for local patterns. I checked naive baselines (2h, 24h, 168h lag) before building the model. The dataset has strong daily and weekly repetition, so the baselines are already competitive. The LSTM does not add much over a simple lag predictor on this data.

**Notebooks:** `Prepro.ipynb` and `keras.ipynb`. There is also `steel_lstm.ipynb` which appears to be an earlier experiment on a different dataset.

---

## CNN_Times_series

**Dataset:** UCI Occupancy Detection. Binary classification: is a room occupied or not. Three files covering different date ranges in February 2015.

**What the data is:** Sensor readings from a single office room: temperature, humidity, light, CO2, and humidity ratio. Light is very strongly correlated with occupancy, which makes this dataset easier than it looks.

**Model:** 1D CNN using sliding windows over the time series. I trained it in both Keras and PyTorch.

**Note from the notebook:** A Random Forest on the raw features already scores well with no windowing and no tuning. The CNN adds complexity without a clear gain on this particular dataset, mainly because light is essentially the answer key.

**Notebooks:** `Prepro.ipynb`, `CNN.keras.ipynb`, and `Pytorch.ipynb`.

---

## Transformer_Network

This is not a training notebook. It is a from-scratch implementation of the transformer attention mechanism using only NumPy, meant to understand how it works internally.

**What it covers:**
- Single-head self-attention: token embeddings, Q/K/V weight matrices, attention scores, scaling by sqrt(d), softmax weights, attention output
- Multi-head self-attention: splitting the embedding dimension across heads, projecting each head separately, concatenating and projecting the output
- Masking

There is also a `Transformer_Architecture.pdf` included as a reference.

**Notebook:** `transformer.ipynb`

