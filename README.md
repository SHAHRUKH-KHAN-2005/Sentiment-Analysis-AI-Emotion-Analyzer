Sentiment Analysis – AI Emotion Analyzer

Created by SHAHRUKH KHAN

An interactive machine-learning web application that analyzes a sentence or review and predicts the emotion expressed in the text. The dashboard provides emotion predictions, prediction history, and visual emotion analytics through a Streamlit interface.

Overview

This project uses Natural Language Processing (NLP) and a trained Support Vector Machine (SVM) classifier to categorize text into one of six emotion classes. Text is converted into numerical features using a saved TF-IDF vectorizer before being passed to the trained model.

The application is designed to make text-based emotion analysis simple and interactive.

Supported Emotion Categories

Emotion

Description

😊 Joy

Happiness, pleasure, or enjoyment

😢 Sadness

Sadness, disappointment, or unhappiness

😠 Anger

Anger, frustration, or irritation

💗 Love

Affection, love, or appreciation

😨 Fear

Fear, worry, or nervousness

😲 Surprise

Surprise or unexpected excitement

Features

Emotion prediction: Enter a sentence, review, or other text and view the predicted emotion.

Sample text/reviews: Explore the analyzer with example text provided by the application.

Six emotion classes: Classifies text into Joy, Sadness, Anger, Love, Fear, or Surprise.

Prediction history: Review predictions made during the current Streamlit session.

CSV export: Download the available prediction history as a CSV file.

Clear history: Clear the recorded prediction history for the current session.

Emotion analytics: Explore the distribution of predicted emotions with visual charts, including bar and doughnut-style charts.

Dark and light themes: Switch the dashboard appearance.

Interactive dashboard: Navigate between Dashboard, Prediction History, Emotion Analytics, and About Project.

Responsive workflow: Analyze text and view the result from one application.

Machine Learning Workflow

The application follows this prediction pipeline:

The user enters a sentence or selects/generates sample text.

The saved TF-IDF vectorizer converts the text into numerical features.

The trained Support Vector Machine (SVM) model predicts the emotion.

The predicted emotion is displayed in the dashboard.

The prediction is added to the current session's history and reflected in the analytics.

Technology Stack

Technology

Purpose

Python

Core programming language

Streamlit

Interactive web application and dashboard

Scikit-learn

TF-IDF text vectorization and SVM machine-learning model

Pandas

Prediction-history and tabular data handling

NumPy

Numerical operations

Plotly / visualization library

Interactive emotion analytics charts

The dashboard shown in the project demonstration identifies Support Vector Machine (SVM) as the classification algorithm and TF-IDF as the text feature representation.

Project Interface

The application includes the following sections:

1. Dashboard

Enter or generate text, run an emotion prediction, and view the result.

2. Prediction History

View predictions recorded during the current session. Export the history to CSV or clear it.

3. Emotion Analytics

View charts showing the distribution and counts of predicted emotion categories.

4. About Project

Read an overview of the project, its model pipeline, and supported emotion classes.

The dashboard also provides a dark/light appearance setting and displays the project credit: Created by SHAHRUKH KHAN.

Getting Started

Prerequisites

Python 3.9 or a compatible Python version

pip

The project source code

The trained SVM model and fitted TF-IDF vectorizer files used by the application

Installation

Clone or download this repository:

git clone https://github.com/SHAHRUKH-KHAN-2005/Sentiment-Analysis-AI-Emotion-Analyzer

cd AI and ML Projects

(Recommended) Create and activate a virtual environment:

Windows

python -m venv venv
venv\Scripts\activate

macOS / Linux

python -m venv venv
source venv/bin/activate

Install the dependencies:

If the repository contains a requirements.txt file:

pip install -r requirements.txt

Otherwise, install the packages used by the application:

pip install streamlit scikit-learn pandas numpy plotly

Make sure the trained model and TF-IDF vectorizer files are present at the paths expected by the application. These fitted artifacts are required for prediction; installing the libraries alone does not recreate them.

Run the Application

From the project directory, run:

streamlit run app.py

Streamlit will provide a local URL in the terminal. Open that URL in your browser to use the application.

If your main Python file has a different name, replace app.py with that filename.

Example

Input:

I am so happy and excited about my new opportunity.

The application returns one of its supported emotion labels. The predicted result depends on the trained model and its learned patterns.

Try different kinds of sentences to explore how the model responds.

Important Notes

This is an emotion classification project. It predicts one label from the six supported categories; it is not a clinical or psychological assessment.

A model prediction is not guaranteed to represent the writer's actual feelings. Context, sarcasm, mixed emotions, spelling, and language can affect results.

Prediction history shown in the dashboard is session-based as demonstrated in the application. It should not be assumed to persist after the session ends.

The README does not claim a particular accuracy or dataset because those details are not established by the demonstration. Add verified dataset information and evaluation results if you want to publish them.

Future Improvements

Add model evaluation metrics such as accuracy, precision, recall, and F1-score.

Support multilingual text.

Improve handling of sarcasm, negation, and mixed emotions.

Add persistent storage for prediction history.

Compare SVM with Logistic Regression, Naive Bayes, and other text-classification models.

Add tests and input validation for more robust predictions.

Author

SHAHRUKH KHAN

Machine Learning | Natural Language Processing | Python

If you find this project useful, consider giving the repository a ⭐ on GitHub.