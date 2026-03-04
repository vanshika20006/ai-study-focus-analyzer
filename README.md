# AI Study Focus Analyzer

> Predict your study focus score, visualize your habits, and get personalized study advice — all running locally with no external APIs.

---

## Overview

The AI Study Focus Analyzer is a machine learning–powered productivity tool built for students. Enter your study session details and instantly receive a predicted focus score, behavioral insights, and tailored recommendations from an on-device NLP chatbot.

---

## Features

**Focus Score Prediction**  
A Random Forest Regression model estimates your focus level (0–100) based on four inputs: study time, phone usage, break count, and mood level.

**AI Study Assistant**  
An intent-based chatbot answers natural language questions like *"How can I improve my focus?"* or *"Why is my score low?"* — powered by TF-IDF vectorization and Logistic Regression, fully offline.

**Study Analytics Dashboard**  
Visual breakdowns of your session data via bar charts (study activity) and pie charts (focus vs. distraction ratio).

---

## Tech Stack

| Layer | Technology |
|---|---|
| App framework | Streamlit |
| ML model | Scikit-learn (Random Forest) |
| NLP pipeline | TF-IDF + Logistic Regression |
| Visualization | Matplotlib |
| Model storage | Joblib |
| Language | Python |

---

## Project Structure

```
ai-study-focus-analyzer/
├── app.py                  # Main Streamlit application
├── train_model.py          # Focus prediction model training
├── requirements.txt
├── bot/
│   ├── chatbot.py          # NLP chatbot logic
│   └── train_nlp.py        # Intent classifier training
├── model/
│   ├── focus_model.pkl
│   ├── nlp_model.pkl
│   └── vectorizer.pkl
├── data/
│   └── study_data.csv
└── utils/
    └── preprocess.py
```

---

## Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/vanshika20006/ai-study-focus-analyzer.git
cd ai-study-focus-analyzer
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the app**
```bash
streamlit run app.py
```

---

## How It Works

### Focus Prediction

The model takes four numeric inputs and returns a score from 0 to 100:

| Input | Description |
|---|---|
| Study Time | Hours spent studying |
| Phone Usage | Time spent on phone (minutes) |
| Break Count | Number of breaks taken |
| Mood Level | Self-reported mood (1–10 scale) |

### NLP Chatbot Pipeline

```
User Input → TF-IDF Vectorization → Logistic Regression → Intent → Response
```

Supported intents: `greeting`, `improve_focus`, `study_tips`, `focus_score`, `low_focus`

---

## Deployment

The app can be deployed for free on [Streamlit Cloud](https://streamlit.io/cloud):

1. Push the project to a public GitHub repository
2. Sign in to Streamlit Cloud and click **New app**
3. Select your repository and set the main file to `app.py`
4. Click **Deploy**

Live demo: `https://ai-study-focus-analyzer.streamlit.app`

---

## Roadmap

- [ ] Pomodoro timer integration
- [ ] Focus trend tracking across sessions
- [ ] Personalized AI study planner
- [ ] Transformer-based NLP (e.g., DistilBERT)
- [ ] ChatGPT-style conversational interface

---

## Author

**Vanshika Sharma**  
B.Tech + M.Tech IT — International Institute of Professional Studies, DAVV  
[GitHub](https://github.com/vanshika20006)