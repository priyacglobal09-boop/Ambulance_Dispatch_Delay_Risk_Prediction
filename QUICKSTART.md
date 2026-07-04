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
/home/priya_paul/.venv/bin/python src/main.py
```

This regenerates data, trains all three models, saves model artifacts, and refreshes figures and reports.

## Optional: open the notebook

```bash
cd notebooks
/home/priya_paul/.venv/bin/jupyter notebook exploration_modeling.ipynb
```

## Troubleshooting

- If `python` is not found, use `/home/priya_paul/.venv/bin/python`.
- If Streamlit cannot find model files, run `src/main.py` once from the repo root.
- If package imports fail, reinstall dependencies with `/home/priya_paul/.venv/bin/pip install -r requirements.txt`.
