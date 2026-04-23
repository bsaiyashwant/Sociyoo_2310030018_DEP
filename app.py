"""
Sociyoo — Social Media Brand Sentiment Analysis Dashboard

A Flask-powered web application that uses ensemble machine learning (Logistic Regression,
Multinomial Naive Bayes, Support Vector Machine) to classify social media posts as
positive, neutral or negative for brand reputation monitoring across Instagram, Twitter,
WhatsApp, Facebook and Telegram.
"""

import logging
import os
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from collections import Counter
import json
import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PLATFORMS = ["instagram", "twitter", "whatsapp", "facebook", "telegram"]
SENTIMENT_LABELS = ["positive", "neutral", "negative"]


# ---------------------------------------------------------------------------
# Training Data
# ---------------------------------------------------------------------------
def build_training_data():
    """
    Build a synthetic labelled corpus of social media posts for three brands
    across all supported platforms.

    Each base sample is replicated across every platform to give the TF-IDF
    vectorizer platform-aware tokens.

    Returns:
        tuple: (texts, labels) where texts is a list of combined
               'platform brand message' strings and labels is the sentiment class.
    """
    base = [
        # BrandA — positive
        ("BrandA", "I love the new update, it works perfectly", "positive"),
        ("BrandA", "Customer service was helpful and quick", "positive"),
        ("BrandA", "The product quality is amazing", "positive"),
        ("BrandA", "Love the new dark mode on the app", "positive"),
        ("BrandA", "Great job fixing the crash bug so quickly", "positive"),
        ("BrandA", "Really impressed with the new camera features", "positive"),
        ("BrandA", "The redesigned homepage looks clean and modern", "positive"),
        ("BrandA", "Voice call quality has improved a lot", "positive"),
        # BrandA — neutral
        ("BrandA", "Not bad, but could be better", "neutral"),
        ("BrandA", "The new design is okay, nothing special", "neutral"),
        ("BrandA", "The story feature is smooth but drains battery", "neutral"),
        ("BrandA", "It works fine for what it is", "neutral"),
        # BrandA — negative
        ("BrandA", "App keeps crashing after the latest update", "negative"),
        ("BrandA", "Very disappointed with the battery life", "negative"),
        ("BrandA", "This update broke my notifications, very annoying", "negative"),
        ("BrandA", "Terrible lag when scrolling through my feed", "negative"),
        ("BrandA", "Privacy settings are confusing and buried", "negative"),
        # BrandB — positive
        ("BrandB", "Just fixed many bugs, great job", "positive"),
        ("BrandB", "Fast delivery and smooth experience", "positive"),
        ("BrandB", "Customer support replied within minutes, great service", "positive"),
        ("BrandB", "Surprised me with excellent support", "positive"),
        ("BrandB", "The checkout process is incredibly seamless", "positive"),
        # BrandB — neutral
        ("BrandB", "Is fine, does what it promises", "neutral"),
        ("BrandB", "Average experience, nothing impressive", "neutral"),
        ("BrandB", "Decent experience, nothing too special", "neutral"),
        ("BrandB", "The app is functional but not exciting", "neutral"),
        # BrandB — negative
        ("BrandB", "Support team ignored my problem", "negative"),
        ("BrandB", "Worst purchase I made this year", "negative"),
        ("BrandB", "The UI looks old compared to competitors", "negative"),
        ("BrandB", "The app freezes whenever I open notifications", "negative"),
        ("BrandB", "Timeline is full of ads, not happy with this", "negative"),
        # BrandC — positive
        ("BrandC", "Listens to customer feedback", "positive"),
        ("BrandC", "Happy with the performance so far", "positive"),
        ("BrandC", "Streaming quality is crystal clear now", "positive"),
        ("BrandC", "Great performance after the last patch", "positive"),
        ("BrandC", "Nice new privacy options, feels safer now", "positive"),
        # BrandC — neutral
        ("BrandC", "Service was neither good nor bad", "neutral"),
        ("BrandC", "Overall experience is okay for the price", "neutral"),
        ("BrandC", "Overall experience is acceptable", "neutral"),
        # BrandC — negative
        ("BrandC", "The app is slow and unresponsive", "negative"),
        ("BrandC", "Features are confusing and hard to use", "negative"),
        ("BrandC", "Keeps logging me out randomly", "negative"),
        ("BrandC", "Stories take ages to load on mobile data", "negative"),
        ("BrandC", "Messages are delayed, very frustrating", "negative"),
        ("BrandC", "Still facing bugs and slow response", "negative"),
    ]
    texts = []
    labels = []
    for platform in PLATFORMS:
        for brand, text, label in base:
            combined = f"{platform} {brand} {text}"
            texts.append(combined)
            labels.append(label)
    logger.info(
        "Training data built: %d samples (%d base × %d platforms)",
        len(texts), len(base), len(PLATFORMS),
    )
    return texts, labels


# ---------------------------------------------------------------------------
# Model Building
# ---------------------------------------------------------------------------
def build_models(texts, labels):
    """
    Train three classifiers on the TF-IDF transformed training corpus.

    Models:
        - Logistic Regression (max_iter=1000)
        - Multinomial Naive Bayes
        - Support Vector Machine (linear kernel, probability estimates)

    Returns:
        tuple: (vectorizer, models_dict)
    """
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
    features = vectorizer.fit_transform(texts)

    models = {}

    logistic = LogisticRegression(max_iter=1000)
    logistic.fit(features, labels)
    models["Logistic Regression"] = logistic

    naive_bayes = MultinomialNB()
    naive_bayes.fit(features, labels)
    models["Multinomial Naive Bayes"] = naive_bayes

    svm = SVC(kernel="linear", probability=True)
    svm.fit(features, labels)
    models["Support Vector Machine"] = svm

    logger.info("Trained %d models on %d features", len(models), features.shape[1])
    return vectorizer, models


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_training_metrics(vectorizer, models, texts, labels):
    """
    Compute training-set accuracy for each individual model and for the
    majority-vote ensemble.

    Returns:
        tuple: (ensemble_accuracy, per_model_accuracy_dict)
    """
    features = vectorizer.transform(texts)
    per_model_accuracy = {}
    for name, clf in models.items():
        preds = clf.predict(features)
        correct = sum(1 for true, pred in zip(labels, preds) if true == pred)
        per_model_accuracy[name] = correct / len(labels) if labels else 0.0

    ensemble_correct = 0
    for index in range(len(labels)):
        row = features[index]
        votes = [clf.predict(row)[0] for clf in models.values()]
        counts = Counter(votes)
        final_label = max(counts, key=counts.get)
        if final_label == labels[index]:
            ensemble_correct += 1
    ensemble_accuracy = ensemble_correct / len(labels) if labels else 0.0
    return ensemble_accuracy, per_model_accuracy


def compute_cross_val_scores(vectorizer, models, texts, labels, cv=5):
    """
    Run stratified k-fold cross-validation for each model and return
    the mean CV accuracy.

    Returns:
        dict: {model_name: mean_cv_accuracy}
    """
    features = vectorizer.transform(texts)
    cv_scores = {}
    for name, clf in models.items():
        scores = cross_val_score(clf, features, labels, cv=cv, scoring="accuracy")
        cv_scores[name] = float(np.mean(scores))
        logger.info("CV accuracy for %s: %.1f%%", name, cv_scores[name] * 100)
    return cv_scores


# ---------------------------------------------------------------------------
# Sample Corpus & Aggregation
# ---------------------------------------------------------------------------
def build_sample_corpus():
    """
    Return a curated list of sample social media posts (platform, brand, text)
    used for the pre-built dashboard visualizations.
    """
    samples = [
        ("instagram", "BrandA", "The new app interface is very intuitive and fast"),
        ("twitter", "BrandA", "I am unhappy with the recent changes"),
        ("facebook", "BrandA", "Decent features for the price"),
        ("twitter", "BrandB", "Surprised me with excellent support"),
        ("instagram", "BrandB", "The latest update made things worse"),
        ("whatsapp", "BrandB", "It is okay, nothing outstanding"),
        ("telegram", "BrandC", "Great performance after the last patch"),
        ("facebook", "BrandC", "Still facing bugs and slow response"),
        ("twitter", "BrandC", "Overall experience is acceptable"),
        ("instagram", "BrandA", "Love how smooth the reels are after the update"),
        ("twitter", "BrandB", "Timeline is full of ads, not happy with this"),
        ("whatsapp", "BrandA", "Voice call quality has improved a lot"),
        ("facebook", "BrandB", "The app freezes whenever I open notifications"),
        ("telegram", "BrandC", "Nice new privacy options, feels safer now"),
        ("instagram", "BrandC", "Stories take ages to load on mobile data"),
        ("twitter", "BrandA", "Great job fixing the crash bug so quickly"),
        ("whatsapp", "BrandC", "Messages are delayed, very frustrating"),
        ("telegram", "BrandA", "The new file sharing feature is wonderful"),
        ("facebook", "BrandA", "Can't believe how slow the app has become"),
        ("instagram", "BrandB", "New profile layout is gorgeous"),
        ("whatsapp", "BrandB", "Group calls keep dropping on WiFi"),
        ("telegram", "BrandB", "Bot integration works perfectly now"),
    ]
    return samples


def aggregate_sentiment_by_platform(samples):
    """
    Group sentiment predictions by platform → brand and count positive /
    neutral / negative occurrences for bar charting.
    """
    data = {}
    for platform, brand, text in samples:
        combined = f"{platform} {brand} {text}"
        label, _, _ = predict_all(combined)
        if platform not in data:
            data[platform] = {}
        if brand not in data[platform]:
            data[platform][brand] = {"positive": 0, "neutral": 0, "negative": 0}
        data[platform][brand][label] += 1
    return data


def build_trend_by_platform(samples):
    """
    Create chronologically indexed sentiment scores (-1, 0, +1) per platform
    for trend-line charting.
    """
    label_to_score = {"negative": -1, "neutral": 0, "positive": 1}
    data = {}
    counters = {}
    for platform, brand, text in samples:
        combined = f"{platform} {brand} {text}"
        label, _, _ = predict_all(combined)
        score = label_to_score[label]
        if platform not in data:
            data[platform] = []
            counters[platform] = 0
        counters[platform] += 1
        data[platform].append(
            {"index": counters[platform], "brand": brand, "score": score, "label": label}
        )
    return data


# ---------------------------------------------------------------------------
# Flask Application
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "sociyoo-dev-secret-key-change-in-prod")

# Build models at startup
logger.info("Building training data and models …")
training_texts, training_labels = build_training_data()
vectorizer, models = build_models(training_texts, training_labels)


def predict_all(text):
    """
    Run a text through every trained classifier, collect votes, and return
    the ensemble majority-vote label.

    Args:
        text: Pre-formatted string ('platform brand message').

    Returns:
        tuple: (final_label, per_model_dict, logistic_probabilities)
    """
    features = vectorizer.transform([text])
    per_model = {}
    votes = []
    for name, clf in models.items():
        label = clf.predict(features)[0]
        votes.append(label)
        probabilities = None
        if hasattr(clf, "predict_proba"):
            probability_vector = clf.predict_proba(features)[0]
            classes = clf.classes_
            probabilities = {
                cls_label: float(prob) for cls_label, prob in zip(classes, probability_vector)
            }
        per_model[name] = {"label": label, "probabilities": probabilities}
    counts = Counter(votes)
    final_label = max(counts, key=counts.get)
    logistic_probabilities = per_model.get("Logistic Regression", {}).get("probabilities") or {}
    return final_label, per_model, logistic_probabilities


# Pre-compute dashboard data
logger.info("Computing metrics and building dashboard data …")
training_accuracy, training_accuracy_per_model = compute_training_metrics(
    vectorizer, models, training_texts, training_labels
)
cv_scores = compute_cross_val_scores(vectorizer, models, training_texts, training_labels)

label_distribution = {"positive": 0, "neutral": 0, "negative": 0}
for label in training_labels:
    if label in label_distribution:
        label_distribution[label] += 1

sample_corpus = build_sample_corpus()
brand_summary_by_platform = aggregate_sentiment_by_platform(sample_corpus)
trend_by_platform = build_trend_by_platform(sample_corpus)
logger.info("Application ready.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    """Main dashboard page with interactive sentiment form and charts."""
    user_result = None
    selected_platform = request.args.get("platform", "twitter").strip().lower()
    if selected_platform not in PLATFORMS:
        selected_platform = "twitter"

    if request.method == "POST":
        selected_platform = (
            request.form.get("platform", selected_platform).strip().lower() or selected_platform
        )
        if selected_platform not in PLATFORMS:
            selected_platform = "twitter"

        brand = request.form.get("brand", "").strip()
        text = request.form.get("text", "").strip()

        if text:
            combined = f"{selected_platform} {brand} {text}".strip()
            predicted_label, per_model, label_to_prob = predict_all(combined)
            user_result = {
                "platform": selected_platform,
                "brand": brand or "Unspecified",
                "text": text,
                "predicted_label": predicted_label,
                "probabilities": label_to_prob,
                "per_model": per_model,
            }
            logger.info(
                "Prediction: platform=%s brand=%s label=%s",
                selected_platform, brand or "N/A", predicted_label,
            )

    summary = brand_summary_by_platform.get(selected_platform, {})
    trend = trend_by_platform.get(selected_platform, [])
    return render_template(
        "index.html",
        app_name="Sociyoo",
        platforms=PLATFORMS,
        selected_platform=selected_platform,
        abstract_text=(
            "Sociyoo is a web-based sentiment dashboard that turns raw social media "
            "conversations into measurable signals for brand reputation management. "
            "Instead of manually reading thousands of posts, tweets and messages, "
            "Sociyoo automatically classifies incoming content from Instagram, Twitter (X), "
            "WhatsApp, Facebook and Telegram as positive, neutral or negative. The system "
            "aggregates these predictions to show which brands are trending positively or "
            "negatively on each platform, highlights changes in sentiment over time and "
            "provides model accuracy and class distribution. Such a tool can support use "
            "cases like campaign monitoring, early crisis detection, feature feedback "
            "analysis and competitor comparison, helping decision makers act quickly on "
            "emerging public opinion."
        ),
        keywords=[
            "Social Media",
            "Sentiment Analysis",
            "Brand Reputation",
            "Natural Language Processing",
            "Machine Learning",
            "Opinion Mining",
        ],
        brand_summary=summary,
        trend_data_json=json.dumps(trend),
        training_accuracy=training_accuracy,
        training_samples=len(training_labels),
        label_distribution=label_distribution,
        training_accuracy_per_model=training_accuracy_per_model,
        cv_scores=cv_scores,
        user_result=user_result,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    JSON API endpoint for programmatic sentiment prediction.

    Request body (JSON):
        {
            "platform": "twitter",
            "brand": "BrandA",
            "text": "I love the new update"
        }

    Returns:
        JSON with predicted sentiment, probabilities and per-model breakdown.
    """
    data = request.get_json(silent=True)
    if not data or not data.get("text"):
        return jsonify({"error": "Missing required field: text"}), 400

    platform = data.get("platform", "twitter").strip().lower()
    if platform not in PLATFORMS:
        platform = "twitter"
    brand = data.get("brand", "").strip()
    text = data.get("text", "").strip()

    combined = f"{platform} {brand} {text}".strip()
    predicted_label, per_model, label_to_prob = predict_all(combined)

    logger.info("API prediction: platform=%s brand=%s label=%s", platform, brand or "N/A", predicted_label)

    return jsonify({
        "platform": platform,
        "brand": brand or "Unspecified",
        "text": text,
        "predicted_sentiment": predicted_label,
        "confidence": label_to_prob,
        "per_model": {
            name: {"label": info["label"], "probabilities": info["probabilities"]}
            for name, info in per_model.items()
        },
    })


@app.route("/health")
def health():
    """Health-check endpoint for deployment readiness."""
    return jsonify({
        "status": "healthy",
        "models_loaded": len(models),
        "training_samples": len(training_labels),
    })


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "development") == "development"
    logger.info("Starting Sociyoo on port %d (debug=%s)", port, debug)
    app.run(debug=debug, port=port)
