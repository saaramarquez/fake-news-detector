# AI Fake News Detector — Setup Guide

A machine learning web app that classifies news articles as REAL or FAKE,
using TF-IDF text vectorization + Logistic Regression, served with Streamlit.

## Project structure
```
fake-news-detector/
├── data/
│   ├── Fake.csv        
│   └── True.csv        
├── model/               <- created automatically after training
├── train_model.py       <- trains the model
├── app.py                <- the Streamlit web app
├── requirements.txt
└── README.md
```

## Step 1: Install Python
You need Python 3.9+.
- Windows/Mac: download from https://www.python.org/downloads/ and run the installer
  (on Windows, tick "Add Python to PATH" during install).
- Check it worked by opening a terminal (Command Prompt / Terminal) and running:
  ```
  python --version
  ```
  (On Mac/Linux it may be `python3 --version`.)

## Step 2: Get the dataset
This project uses the popular **"Fake and Real News Dataset"** by Clement Bisaillon on Kaggle:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

1. Go to that link (you'll need a free Kaggle account to download).
2. Download the dataset — it comes as a zip with `Fake.csv` and `True.csv`.
3. Unzip it and place both files inside this project's `data/` folder.

Each CSV has columns: `title`, `text`, `subject`, `date`. The training script only needs `title` and `text`.

## Step 3: Set up a virtual environment (recommended, keeps things clean)
Open a terminal **inside the `fake-news-detector` folder** and run:

```bash
python -m venv venv
```

Activate it:
- Windows: `venv\Scripts\activate`
- Mac/Linux: `source venv/bin/activate`

You'll see `(venv)` appear in your terminal prompt when it's active.

## Step 4: Install the required packages
```bash
pip install -r requirements.txt
```

## Step 5: Train the model
```bash
python train_model.py
```
This will:
- Load and clean the dataset
- Split it into training/test sets
- Convert text into TF-IDF vectors
- Train a Logistic Regression classifier
- Print accuracy and a classification report (should land around 98-99% accuracy on this dataset)
- Save `model/fake_news_model.joblib` and `model/vectorizer.joblib`

## Step 6: Run the web app
```bash
streamlit run app.py
```
This opens a browser tab at `http://localhost:8501`. Paste any news text in and click
"Check Article" to get a REAL/FAKE prediction with a confidence score.
