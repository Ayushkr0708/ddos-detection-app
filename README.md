# Advanced DDoS Detection Streamlit App

This project is a machine-learning based DDoS detection dashboard built with Streamlit. It predicts whether uploaded network traffic records are `BENIGN` or `DDoS` and displays confidence scores, probability curves, risk levels, feature importance, and model metrics.

## Project Structure

```text
ddos-detection-app/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
├── datasets/
│   └── dataset.csv
├── model/
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── features.pkl
│   ├── metrics.pkl
│   └── feature_importance.csv
├── screenshots/
└── docs/
    ├── architecture.png
    ├── workflow.png
    └── project_report.md
```

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Retrain Model

```bash
python train.py --dataset datasets/dataset.csv
```

## Deploy to Streamlit Community Cloud

1. Create a public GitHub repository.
2. Upload all files and folders from this project.
3. Go to https://share.streamlit.io.
4. Click **New app**.
5. Select your repository, branch `main`, and main file path `app.py`.
6. Click **Deploy**.

## Notes

- Do not use Colab/ngrok for a permanent app link.
- Streamlit Cloud gives a persistent public URL.
- Keep model paths as `model/model.pkl`, `model/scaler.pkl`, and `model/features.pkl`.
