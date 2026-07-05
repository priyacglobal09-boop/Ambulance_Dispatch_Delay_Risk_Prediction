# Quick Start

## 1. Go to the repo

```bash
cd /home/priya_paul/.venv/repos/Ambulance_Dispatch_Delay_Risk_Prediction
```

## 2. Install dependencies

```bash
/home/priya_paul/.venv/bin/pip install -r requirements.txt
```

## 3. Run the Streamlit app

```bash
/home/priya_paul/.venv/bin/streamlit run app/streamlit_app.py
```

The app should open at `http://localhost:8501`.

## Optional: retrain everything

```bash
/home/priya_paul/.venv/bin/python src/main.py --data-source current
```

This trains on the current normalized dataset, trains all three models, saves model artifacts, and refreshes figures and reports.

## Optional: train from Kaggle IERAD

Configure Kaggle credentials at `~/.kaggle/kaggle.json`, then run:

```bash
/home/priya_paul/.venv/bin/python src/main.py --data-source ierad --download-ierad
```

If you downloaded the dataset manually:

```bash
/home/priya_paul/.venv/bin/python src/main.py --data-source ierad --ierad-input /path/to/ierad.csv
```

For faster local iteration on the full Kaggle file:

```bash
/home/priya_paul/.venv/bin/python src/main.py --data-source ierad --ierad-input /path/to/ierad.csv --max-records 30000 --integer-epochs 10 --embedding-epochs 12
```

For the current higher-quality artifact build:

```bash
/home/priya_paul/.venv/bin/python src/main.py --data-source current --max-records 100000 --integer-epochs 15 --embedding-epochs 20
```

To regenerate the offline synthetic fallback dataset:

```bash
/home/priya_paul/.venv/bin/python src/main.py --data-source synthetic
```

## Optional: open the notebook

```bash
cd notebooks
/home/priya_paul/.venv/bin/jupyter notebook exploration_modeling.ipynb
```

## Troubleshooting

- If `python` is not found, use `/home/priya_paul/.venv/bin/python`.
- If Streamlit cannot find model files, run `src/main.py` once from the repo root.
- If package imports fail, reinstall dependencies with `/home/priya_paul/.venv/bin/pip install -r requirements.txt`.
