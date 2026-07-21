# ============================================================
# AUTO EMAIL / TICKET CATEGORIZER - IMPROVED VERSION
# Categories: Billing, Technical, HR, General
# ============================================================

import re
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ------------------------------------------------------------
# 1. LARGER BALANCED DATASET
# ------------------------------------------------------------

data = {
    "subject": [

        # BILLING - 20
        "Invoice not received",
        "Refund is still pending",
        "Incorrect amount charged",
        "Payment failed",
        "Need billing receipt",
        "Subscription charged twice",
        "Money deducted but order failed",
        "Unexpected subscription fee",
        "Need duplicate invoice",
        "Refund amount is incorrect",
        "Card charged without purchase",
        "Billing address update",
        "Payment confirmation needed",
        "Cancel recurring payment",
        "Discount not applied",
        "Tax amount incorrect",
        "Invoice contains wrong details",
        "Payment pending for many days",
        "Need monthly billing statement",
        "Amount debited twice",

        # TECHNICAL - 20
        "Unable to login",
        "Application crashes",
        "Website is not loading",
        "Password reset not working",
        "API returns server error",
        "Software installation problem",
        "Mobile app is frozen",
        "Database connection failed",
        "Page shows blank screen",
        "Unable to upload document",
        "OTP not received",
        "System is very slow",
        "File download is failing",
        "Account login error",
        "Application shows error code",
        "Server is unavailable",
        "Unable to access dashboard",
        "Notification feature not working",
        "Browser compatibility problem",
        "App closes automatically",

        # HR - 20
        "Leave balance incorrect",
        "Salary slip required",
        "Request for experience letter",
        "Update employee details",
        "Question about company holidays",
        "Need HR policy document",
        "Maternity leave policy",
        "Provident fund information",
        "Need employment verification",
        "Request relieving letter",
        "Attendance record incorrect",
        "Need offer letter copy",
        "Update bank account details",
        "Question about probation period",
        "Need income tax document",
        "Request work from home",
        "Employee ID card required",
        "Need appraisal details",
        "Question about notice period",
        "Request casual leave",

        # GENERAL - 20
        "Office address enquiry",
        "Product information required",
        "How can I contact support",
        "Request for company brochure",
        "Need general assistance",
        "Working hours enquiry",
        "Company location details",
        "Contact customer service",
        "Need information about services",
        "Business partnership enquiry",
        "Where is your branch located",
        "Need company phone number",
        "General feedback",
        "Request product catalogue",
        "How to reach your office",
        "Need website information",
        "Question about company",
        "Service availability enquiry",
        "Need sales team contact",
        "Request general information"
    ],

    "body": [

        # BILLING
        "I have not received the invoice for my recent payment.",
        "My refund was approved but the amount has not reached my account.",
        "The amount shown on my bill is incorrect.",
        "I am unable to complete the online payment.",
        "Please send me a receipt for the payment made yesterday.",
        "I was charged two times for the same subscription.",
        "Money was deducted but the order was not completed.",
        "I noticed an unknown subscription charge in my account.",
        "Please send another copy of my invoice.",
        "The refund credited is less than the expected amount.",
        "My card was charged even though I did not make a purchase.",
        "I want to change the address printed on my invoice.",
        "Please confirm whether my payment was received.",
        "Please stop the automatic recurring subscription payment.",
        "The promotional discount was not applied to my bill.",
        "The tax charged on the invoice appears to be wrong.",
        "My company name and address are incorrect on the invoice.",
        "The payment status has remained pending for several days.",
        "Please send the billing statement for this month.",
        "The same amount was debited twice from my bank account.",

        # TECHNICAL
        "I cannot access my account even with the correct password.",
        "The mobile application closes immediately after opening.",
        "The company website is unavailable and displays an error.",
        "The password reset link is not opening.",
        "The API is returning a 500 internal server error.",
        "I am unable to install the software on my computer.",
        "The mobile application is frozen and does not respond.",
        "The application cannot connect to the database.",
        "The page loads but only displays a blank screen.",
        "I cannot upload my document to the portal.",
        "The login OTP is not arriving on my phone.",
        "The system takes several minutes to load every page.",
        "The download stops before the file is completed.",
        "I receive an authentication error while logging in.",
        "The application repeatedly displays an unknown error code.",
        "The server is down and users cannot access the service.",
        "I cannot open the dashboard after signing in.",
        "Push notifications are not appearing in the application.",
        "The website does not work correctly in my browser.",
        "The application closes automatically whenever I open it.",

        # HR
        "My available leave balance is showing the wrong value.",
        "Please send my salary slip for this month.",
        "I need an experience certificate from the company.",
        "Please change my phone number in the employee profile.",
        "Can you provide the list of company holidays?",
        "Please share the latest employee leave policy.",
        "I want to know the rules for maternity leave.",
        "Please provide details regarding my provident fund account.",
        "I need an employment verification letter for a bank.",
        "Please issue my relieving letter after resignation.",
        "My attendance data is incorrect for last week.",
        "Please send a copy of my original offer letter.",
        "I want to update my salary bank account information.",
        "How long is the probation period for new employees?",
        "Please provide my Form 16 or tax document.",
        "I would like to request work from home for two days.",
        "My employee identification card has not been issued.",
        "Please explain the employee appraisal process.",
        "What is the notice period mentioned in company policy?",
        "I want to apply for one day of casual leave.",

        # GENERAL
        "Could you provide the location of your main office?",
        "I would like more details about your products.",
        "Please tell me how to contact the customer support team.",
        "I need a company brochure for general information.",
        "I have a general question and need assistance.",
        "What are the normal office working hours?",
        "Please provide the complete address of the company.",
        "How can I speak with a customer service representative?",
        "Please send information about the services you provide.",
        "I would like to discuss a possible business partnership.",
        "Where is your nearest branch office located?",
        "Please provide the official company contact number.",
        "I would like to submit feedback about your service.",
        "Please send the latest catalogue of your products.",
        "Can you provide directions to your office?",
        "I need general information about your website.",
        "I would like to know more about your company.",
        "Are your services available in my city?",
        "Please share the contact details of the sales department.",
        "I need some general information before purchasing."
    ],

    "category": (
        ["Billing"] * 20 +
        ["Technical"] * 20 +
        ["HR"] * 20 +
        ["General"] * 20
    )
}

df = pd.DataFrame(data)


# ------------------------------------------------------------
# 2. TEXT CLEANING
# ------------------------------------------------------------

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


df["text"] = (
    df["subject"].fillna("") + " " +
    df["body"].fillna("")
)

df["text"] = df["text"].apply(clean_text)

print("Dataset shape:", df.shape)

print("\nCategory counts:")
print(df["category"].value_counts())


# ------------------------------------------------------------
# 3. TRAIN-TEST SPLIT
# ------------------------------------------------------------

X = df["text"]
y = df["category"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# ------------------------------------------------------------
# 4. IMPROVED MODEL
# ------------------------------------------------------------

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=10000,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )
    )
])


# ------------------------------------------------------------
# 5. TRAIN MODEL
# ------------------------------------------------------------

model.fit(X_train, y_train)

print("\nModel training completed.")


# ------------------------------------------------------------
# 6. MODEL EVALUATION
# ------------------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n================ MODEL EVALUATION ================")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ------------------------------------------------------------
# 7. CROSS-VALIDATION
# ------------------------------------------------------------

cross_validation_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("\nCross-validation scores:")
print(cross_validation_scores)

print(
    f"Average cross-validation accuracy: "
    f"{cross_validation_scores.mean() * 100:.2f}%"
)


# ------------------------------------------------------------
# 8. PRIORITY DETECTION
# ------------------------------------------------------------

def detect_priority(ticket_text):

    urgent_keywords = [
        "urgent",
        "immediately",
        "critical",
        "server down",
        "website down",
        "not working",
        "failed",
        "crash",
        "crashes",
        "error",
        "unable to access",
        "security issue",
        "payment failed",
        "money deducted",
        "charged twice",
        "account blocked"
    ]

    cleaned_ticket = clean_text(ticket_text)

    for keyword in urgent_keywords:
        if keyword in cleaned_ticket:
            return "Urgent"

    return "Normal"


# ------------------------------------------------------------
# 9. PREDICTION FUNCTION
# ------------------------------------------------------------

def categorize_ticket(subject, body, confidence_threshold=0.50):

    combined_text = clean_text(subject + " " + body)

    probabilities = model.predict_proba([combined_text])[0]

    predicted_index = probabilities.argmax()
    predicted_category = model.classes_[predicted_index]
    confidence = probabilities[predicted_index]

    priority = detect_priority(combined_text)

    if confidence < confidence_threshold:
        final_route = "Needs Human Review"
    else:
        final_route = predicted_category

    return {
        "subject": subject,
        "predicted_category": predicted_category,
        "confidence": round(confidence * 100, 2),
        "priority": priority,
        "final_route": final_route
    }


# ------------------------------------------------------------
# 10. TEST ON FIVE NEW TICKETS
# ------------------------------------------------------------

new_tickets = [
    {
        "subject": "Payment deducted twice",
        "body": "The same amount was debited two times from my bank account."
    },
    {
        "subject": "Urgent application issue",
        "body": "The mobile application crashes whenever I try to login."
    },
    {
        "subject": "Leave policy question",
        "body": "Please share the maternity leave policy."
    },
    {
        "subject": "Need office location",
        "body": "Can you send the address of your Bangalore office?"
    },
    {
        "subject": "Password reset problem",
        "body": "The reset password link is not working."
    }
]

print("\n================ NEW TICKET PREDICTIONS ================")

prediction_results = []

for number, ticket in enumerate(new_tickets, start=1):

    result = categorize_ticket(
        ticket["subject"],
        ticket["body"]
    )

    prediction_results.append(result)

    print(f"\nTicket {number}")
    print(f"Subject            : {result['subject']}")
    print(f"Predicted category : {result['predicted_category']}")
    print(f"Confidence         : {result['confidence']}%")
    print(f"Priority           : {result['priority']}")
    print(f"Final route        : {result['final_route']}")


# ------------------------------------------------------------
# 11. SAVE OUTPUTS
# ------------------------------------------------------------

results_df = pd.DataFrame(prediction_results)

results_df.to_csv(
    "ticket_predictions.csv",
    index=False
)

joblib.dump(
    model,
    "ticket_categorizer_model.pkl"
)

print("\nPrediction results saved as ticket_predictions.csv")
print("Model saved as ticket_categorizer_model.pkl")


# ------------------------------------------------------------
# 12. LIVE TICKET DEMO
# ------------------------------------------------------------

print("\n================ LIVE TICKET DEMO ================")

while True:

    subject = input(
        "\nEnter ticket subject or type 'exit' to stop: "
    ).strip()

    if subject.lower() == "exit":
        print("Ticket categorizer closed.")
        break

    body = input("Enter ticket body: ").strip()

    result = categorize_ticket(subject, body)

    print("\n---------- ROUTING RESULT ----------")
    print(f"Category    : {result['predicted_category']}")
    print(f"Confidence  : {result['confidence']}%")
    print(f"Priority    : {result['priority']}")
    print(f"Final route : {result['final_route']}")
