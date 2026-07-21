# Ticket-categorizer
A machine learning project that automatically classifies support tickets using NLP techniques, helping route customer requests with confidence scoring, priority detection, and human review for uncertain cases.
# Auto Email / Ticket Categorizer

This project classifies support tickets into:

- Billing
- Technical
- HR
- General

## Approach

The ticket subject and body are combined and cleaned. TF-IDF converts the text into numerical features, and a Multinomial Naive Bayes model predicts the category.

The system also provides:

- Confidence score
- Human review for low-confidence predictions
- Urgent or normal priority tagging
- Predictions for new tickets

## Technologies

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Multinomial Naive Bayes

## Run

```bash
pip install pandas scikit-learn joblib
python ticket_categorizer.py
