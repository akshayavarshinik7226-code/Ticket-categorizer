# Model Evaluation Results

## Dataset

- Total Samples: 80
- Categories:
  - Billing: 20
  - Technical: 20
  - HR: 20
  - General: 20

## Model

- Algorithm: Logistic Regression
- Feature Extraction: TF-IDF
- Validation: 5-Fold Cross Validation

## Performance

- Test Accuracy: **90%**
- Average Cross Validation Accuracy: **85%**

## Evaluation Metrics

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Additional Features

- Confidence Score
- Priority Detection
- Human Review Routing
- Prediction on New Tickets
- Model Serialization (.pkl)

## Sample Predictions

- Payment deducted twice → Billing
- Urgent application issue → Technical
- Leave policy question → HR
- Need office location → General
- Password reset problem → Technical
