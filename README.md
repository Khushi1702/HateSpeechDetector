# HateSpeechDetector

## Overview
HateSpeechDetector is a comprehensive project designed to detect hate speech within Instagram reels and comments. Utilizing advanced Natural Language Processing (NLP) techniques and machine learning algorithms, this project aims to create an effective tool for identifying and mitigating hate speech on social media platforms.

## Features
- Data collection from Instagram reels and comments
- Data preprocessing including text cleaning, tokenization, and stemming
- Implementation of machine learning models: Support Vector Machine (SVM), Logistic Regression, and Decision Tree
- Evaluation of models using accuracy, precision, recall, and F1-score
- Visualization of model performance metrics

## Motivation
The rise of hate speech on social media platforms poses a significant challenge to online safety. This project aims to develop robust mechanisms for detecting and managing hate speech, contributing to a safer and more inclusive online environment.

## Problem Definition
Develop a scalable hate speech detection system for Instagram, capable of handling real-time data and accurately distinguishing between hate speech and non-hate speech content. The system should balance user experience and freedom of expression while promoting a safer online community.

## Objectives
1. Develop accurate machine learning models for hate speech detection.
2. Apply NLP techniques for effective data preprocessing.
3. Achieve precise categorization of hate speech and non-hate speech.
4. Ensure model robustness against the nuances of online offensive language.
5. Provide insights and interpretability of hate speech detection results.

## Tools and Technologies
- **Python**: Primary programming language for data handling and model implementation.
- **Pandas and NLTK**: For data manipulation and NLP tasks.
- **Scikit-learn**: For implementing machine learning models.
- **Matplotlib**: For visualizing model evaluation metrics.
- **Comment Exporter**: Extension for downloading Instagram comments.

## Data Collection and Preprocessing
- **Data Collection**: Using an Instagram comment exporter and audio-to-text transcriber for reels.
- **Preprocessing Steps**:
  - Remove missing values and irrelevant content.
  - Clean text by removing hyperlinks, special characters, and numbers.
  - Tokenize and remove stop words.
  - Apply stemming for word normalization.

## Model Implementation
- **Logistic Regression**: With L1 and L2 regularization.
- **Decision Tree**: For recursive data partitioning.
- **Support Vector Machine (SVM)**: For defining decision boundaries.

## Evaluation Metrics
- **Accuracy**: Measure of correctly identified instances.
- **Precision**: Proportion of true positive results.
- **Recall**: Ability to identify all relevant instances.
- **F1-Score**: Harmonic mean of precision and recall.

## Results
| Model                | Accuracy | Precision | Recall | F1 Score |
|----------------------|----------|-----------|--------|----------|
| SVM                  | 0.80     | 0.7950    | 0.80   | 0.7836   |
| Logistic Regression  | 0.7857   | 0.8095    | 0.7857 | 0.7469   |
| Decision Tree        | 0.7107   | 0.7077    | 0.7107 | 0.7091   |

## Conclusion
The SVM model demonstrated superior performance in detecting hate speech compared to Logistic Regression and Decision Tree models. This project highlights the complexity of distinguishing hate speech from other offensive language and the importance of sophisticated algorithms for accurate detection.

## References
- U. Bhandary, "Detection of Hate Speech in Videos Using Machine Learning," 2019.
- F. Alkomah and X. A. Ma, "Literature Review of Textual Hate Speech Detection Methods and Datasets," 2022.
- T. Davidson et al., "Automated Hate Speech Detection and the Problem of Offensive Language," 2017.
- M. S. A. Sanoussi et al., "Detection of Hate Speech Texts Using Machine Learning Algorithm," 2022.

## List of Abbreviations
- **CSV**: Comma Separated Values
- **SVM**: Support Vector Machine
- **NLP**: Natural Language Processing
- **NLTK**: Natural Language ToolKit
- **URL**: Uniform Resource Locator
- **ML**: Machine Learning
- **TF-IDF**: Term Frequency - Inverse Document Frequency
- **XAI**: Explainable Artificial Intelligence
- **IEEE**: Institute of Electrical and Electronics Engineers
