# 📊 Sociyoo — Social Media Brand Sentiment Analysis

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Chart.js](https://img.shields.io/badge/Chart.js-4.x-FF6384?style=for-the-badge&logo=chartdotjs&logoColor=white)

**An ML-powered sentiment intelligence dashboard for monitoring brand reputation across five social media platforms in real time.**

[Features](#-features) · [Demo](#-demo) · [Quick Start](#-quick-start) · [API](#-api-endpoint) · [Architecture](#-architecture) · [Tech Stack](#-tech-stack) · [License](#-license)

</div>

---

## 🚀 Features

| Feature | Description |
|---------|-------------|
| 🔍 **Interactive Classification** | Paste any social media post and get instant sentiment prediction |
| 📊 **Brand Comparison Charts** | Grouped bar charts comparing positive/neutral/negative sentiment per brand |
| 📈 **Sentiment Trend Lines** | Chronological score tracking to spot reputation shifts early |
| 🤖 **Ensemble ML Pipeline** | Majority-vote across Logistic Regression, Naive Bayes & SVM |
| 📱 **Multi-Platform Support** | Instagram, Twitter (X), WhatsApp, Facebook, Telegram |
| 🎯 **Probability Breakdown** | Visual progress bars for per-class confidence scores |
| 📉 **Model Accuracy Dashboard** | Training accuracy, cross-validation scores & class distribution |
| 🔌 **REST API** | JSON endpoint (`/api/predict`) for programmatic access |
| 🩺 **Health Check** | `/health` endpoint for deployment monitoring |
| 🌙 **Premium Dark UI** | Glassmorphism, animations, responsive design with Inter typography |

---

## 🎯 Demo

### Dashboard Overview
The main dashboard provides an at-a-glance view of brand sentiment across platforms with interactive charts.

### Interactive Classification
Users can select a platform, enter a brand name, and paste a social media message to get real-time sentiment analysis with probability breakdowns and per-algorithm predictions.

### Model Performance
The performance section shows ensemble accuracy, per-algorithm comparison (horizontal bar chart), and training data distribution (doughnut chart).

---

## ⚡ Quick Start

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)

---

## 🧠 ML Pipeline

### Feature Extraction
- **TF-IDF Vectorizer** with uni-grams and bi-grams
- Platform and brand name prepended to text for context-aware features

### Classifiers
| Algorithm | Type | Configuration |
|-----------|------|---------------|
| Logistic Regression | Linear | `max_iter=1000`, probability enabled |
| Multinomial Naive Bayes | Probabilistic | Default priors |
| Support Vector Machine | Kernel-based | `kernel='linear'`, probability enabled |

### Ensemble Strategy
- **Majority Voting**: Each classifier casts one vote; the label with the most votes wins
- Probability estimates from Logistic Regression are surfaced for confidence display

### Evaluation
- Training accuracy computed on full labelled dataset
- 5-fold stratified cross-validation for generalization estimates

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.9+, Flask 3.0 |
| **ML** | Scikit-learn 1.4 (TF-IDF, LogReg, NB, SVM) |
| **Frontend** | HTML5, CSS3 (glassmorphism), Jinja2 |
| **Charts** | Chart.js 4.x |
| **Fonts** | Google Fonts — Inter |
| **Deployment** | Gunicorn (production WSGI) |

---

## 📁 Project Structure

```
brand_sentiment_analysis/
├── app.py                 # Flask app, ML pipeline, routes & API
├── requirements.txt       # Python dependencies
├── LICENSE                # MIT License
├── README.md              # This file
├── static/
│   └── style.css          # Premium dark-mode stylesheet
└── templates/
    └── index.html         # Dashboard template (Jinja2 + Chart.js)
```

---

## 📋 Use Cases

- **Campaign Monitoring** — Track sentiment shifts during marketing campaigns
- **Crisis Detection** — Early warning system for negative sentiment spikes
- **Feature Feedback** — Gauge user reaction to product updates
- **Competitor Comparison** — Side-by-side brand reputation analysis
- **Social Listening** — Aggregate opinions from multiple platforms

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the copyrights to Balivada Sai Yashwant
