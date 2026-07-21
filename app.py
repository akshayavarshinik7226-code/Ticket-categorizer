# ============================================================
# SMART TICKET ROUTER - STREAMLIT APPLICATION
# ============================================================

import re
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st


# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="Smart Ticket Router",
    page_icon="🎫",
    layout="wide"
)


# ------------------------------------------------------------
# LOAD THE TRAINED MODEL
# ------------------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("ticket_categorizer_model.pkl")


try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "Model file not found. Run ticket_categorizer.py first "
        "to create ticket_categorizer_model.pkl."
    )
    st.stop()


# ------------------------------------------------------------
# TEXT CLEANING
# ------------------------------------------------------------

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ------------------------------------------------------------
# PRIORITY DETECTION
# ------------------------------------------------------------

def detect_priority(ticket_text):

    critical_keywords = [
        "security breach",
        "data leak",
        "account hacked",
        "system down",
        "server down",
        "website down",
        "production down"
    ]

    urgent_keywords = [
        "urgent",
        "immediately",
        "critical",
        "not working",
        "failed",
        "crash",
        "crashes",
        "error",
        "unable to access",
        "payment failed",
        "money deducted",
        "charged twice",
        "account blocked",
        "refund pending"
    ]

    cleaned_ticket = clean_text(ticket_text)

    matched_critical = [
        word for word in critical_keywords
        if word in cleaned_ticket
    ]

    matched_urgent = [
        word for word in urgent_keywords
        if word in cleaned_ticket
    ]

    if matched_critical:
        return "Critical", matched_critical

    if matched_urgent:
        return "Urgent", matched_urgent

    return "Normal", []


# ------------------------------------------------------------
# SLA RECOMMENDATION
# ------------------------------------------------------------

def recommend_sla(priority):

    sla_mapping = {
        "Critical": "Respond within 15 minutes",
        "Urgent": "Respond within 1 hour",
        "Normal": "Respond within 8 business hours"
    }

    return sla_mapping.get(
        priority,
        "Respond within 8 business hours"
    )


# ------------------------------------------------------------
# TEAM ROUTING INFORMATION
# ------------------------------------------------------------

def get_team_details(category):

    team_mapping = {
        "Billing": {
            "team": "Finance & Billing Team",
            "action": "Verify transaction, invoice or refund details."
        },
        "Technical": {
            "team": "Technical Support Team",
            "action": "Review logs, reproduce the issue and troubleshoot."
        },
        "HR": {
            "team": "Human Resources Team",
            "action": "Verify employee records and applicable HR policy."
        },
        "General": {
            "team": "Customer Support Team",
            "action": "Review the enquiry and provide relevant information."
        }
    }

    return team_mapping.get(
        category,
        {
            "team": "Manual Review Team",
            "action": "Review and assign the ticket manually."
        }
    )


# ------------------------------------------------------------
# PREDICTION FUNCTION
# ------------------------------------------------------------

def predict_ticket(subject, body, confidence_threshold):

    combined_text = clean_text(
        f"{subject} {body}"
    )

    probabilities = model.predict_proba(
        [combined_text]
    )[0]

    classes = model.classes_

    probability_table = pd.DataFrame({
        "Category": classes,
        "Probability": probabilities
    }).sort_values(
        by="Probability",
        ascending=False
    )

    predicted_index = probabilities.argmax()
    predicted_category = classes[predicted_index]
    confidence = float(probabilities[predicted_index])

    priority, matched_keywords = detect_priority(
        combined_text
    )

    if confidence < confidence_threshold:
        final_route = "Needs Human Review"
        assigned_team = "Manual Review Team"
    else:
        final_route = predicted_category
        assigned_team = get_team_details(
            predicted_category
        )["team"]

    return {
        "subject": subject,
        "body": body,
        "predicted_category": predicted_category,
        "confidence": confidence,
        "priority": priority,
        "matched_keywords": matched_keywords,
        "final_route": final_route,
        "assigned_team": assigned_team,
        "sla": recommend_sla(priority),
        "probability_table": probability_table
    }


# ------------------------------------------------------------
# SESSION HISTORY
# ------------------------------------------------------------

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------

st.title("🎫 Smart Ticket Router")

st.write(
    "An NLP-powered helpdesk triage system that categorizes "
    "incoming support tickets, estimates confidence, detects "
    "priority and routes uncertain cases for human review."
)

st.divider()


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

with st.sidebar:

    st.header("Model Controls")

    confidence_threshold = st.slider(
        "Human-review confidence threshold",
        min_value=0.30,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help=(
            "Tickets below this confidence are routed "
            "to the manual-review queue."
        )
    )

    st.info(
        "Model: TF-IDF + Logistic Regression\n\n"
        "Categories: Billing, Technical, HR and General"
    )

    st.metric(
        label="Recorded model accuracy",
        value="90%"
    )

    st.caption(
        "Accuracy is based on the current held-out test set."
    )


# ------------------------------------------------------------
# MAIN TABS
# ------------------------------------------------------------

single_tab, batch_tab, history_tab, model_tab = st.tabs([
    "Single Ticket",
    "Batch Prediction",
    "Prediction History",
    "Model Information"
])


# ------------------------------------------------------------
# SINGLE-TICKET TAB
# ------------------------------------------------------------

with single_tab:

    st.subheader("Classify a New Ticket")

    subject = st.text_input(
        "Ticket subject",
        placeholder="Example: Payment deducted twice"
    )

    body = st.text_area(
        "Ticket description",
        placeholder=(
            "Example: The amount was debited twice from "
            "my bank account."
        ),
        height=160
    )

    classify_button = st.button(
        "Classify Ticket",
        type="primary",
        use_container_width=True
    )

    if classify_button:

        if not subject.strip() and not body.strip():
            st.warning(
                "Enter a subject or ticket description."
            )

        else:
            result = predict_ticket(
                subject,
                body,
                confidence_threshold
            )

            history_entry = {
                "Timestamp": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "Subject": subject,
                "Predicted Category": (
                    result["predicted_category"]
                ),
                "Confidence": round(
                    result["confidence"] * 100,
                    2
                ),
                "Priority": result["priority"],
                "Final Route": result["final_route"],
                "Assigned Team": result["assigned_team"],
                "SLA": result["sla"]
            }

            st.session_state.prediction_history.append(
                history_entry
            )

            st.success(
                "Ticket analysed successfully."
            )

            metric_one, metric_two, metric_three = st.columns(3)

            metric_one.metric(
                "Predicted Category",
                result["predicted_category"]
            )

            metric_two.metric(
                "Confidence",
                f"{result['confidence'] * 100:.2f}%"
            )

            metric_three.metric(
                "Priority",
                result["priority"]
            )

            st.progress(
                min(result["confidence"], 1.0),
                text=(
                    f"Model confidence: "
                    f"{result['confidence'] * 100:.2f}%"
                )
            )

            if result["final_route"] == "Needs Human Review":
                st.warning(
                    "Low-confidence ticket: route this case "
                    "to the human-review queue."
                )
            else:
                st.info(
                    f"Recommended route: "
                    f"{result['assigned_team']}"
                )

            route_details = get_team_details(
                result["predicted_category"]
            )

            left_column, right_column = st.columns(2)

            with left_column:
                st.markdown("### Routing recommendation")
                st.write(
                    f"**Final route:** "
                    f"{result['final_route']}"
                )
                st.write(
                    f"**Assigned team:** "
                    f"{result['assigned_team']}"
                )
                st.write(
                    f"**Suggested SLA:** "
                    f"{result['sla']}"
                )
                st.write(
                    f"**Next action:** "
                    f"{route_details['action']}"
                )

            with right_column:
                st.markdown("### Priority explanation")

                if result["matched_keywords"]:
                    st.write(
                        "Priority keywords detected:"
                    )

                    for keyword in result["matched_keywords"]:
                        st.code(keyword)
                else:
                    st.write(
                        "No urgent or critical keywords "
                        "were detected."
                    )

            st.markdown("### Category probability breakdown")

            probability_display = (
                result["probability_table"].copy()
            )

            probability_display["Probability"] = (
                probability_display["Probability"] * 100
            ).round(2)

            st.bar_chart(
                probability_display.set_index(
                    "Category"
                )
            )

            st.dataframe(
                probability_display,
                use_container_width=True,
                hide_index=True
            )


# ------------------------------------------------------------
# BATCH-PREDICTION TAB
# ------------------------------------------------------------

with batch_tab:

    st.subheader("Batch Ticket Classification")

    st.write(
        "Upload a CSV containing `subject` and `body` columns."
    )

    sample_data = pd.DataFrame({
        "subject": [
            "Refund pending",
            "Application crashes",
            "Need salary slip"
        ],
        "body": [
            "My refund has not reached my account.",
            "The app closes whenever I open it.",
            "Please send my salary slip for June."
        ]
    })

    sample_csv = sample_data.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Sample CSV",
        data=sample_csv,
        file_name="sample_tickets.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader(
        "Upload ticket CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            uploaded_df = pd.read_csv(
                uploaded_file
            )

            required_columns = {
                "subject",
                "body"
            }

            if not required_columns.issubset(
                uploaded_df.columns
            ):
                st.error(
                    "CSV must contain `subject` and `body` columns."
                )

            else:
                batch_results = []

                for _, row in uploaded_df.iterrows():

                    result = predict_ticket(
                        str(row["subject"]),
                        str(row["body"]),
                        confidence_threshold
                    )

                    batch_results.append({
                        "subject": row["subject"],
                        "body": row["body"],
                        "predicted_category": (
                            result["predicted_category"]
                        ),
                        "confidence_percent": round(
                            result["confidence"] * 100,
                            2
                        ),
                        "priority": result["priority"],
                        "final_route": result["final_route"],
                        "assigned_team": (
                            result["assigned_team"]
                        ),
                        "sla": result["sla"]
                    })

                batch_results_df = pd.DataFrame(
                    batch_results
                )

                st.dataframe(
                    batch_results_df,
                    use_container_width=True,
                    hide_index=True
                )

                batch_csv = batch_results_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "Download Classified Results",
                    data=batch_csv,
                    file_name="classified_tickets.csv",
                    mime="text/csv",
                    type="primary"
                )

        except Exception as error:
            st.error(
                f"Unable to process the uploaded file: {error}"
            )


# ------------------------------------------------------------
# HISTORY TAB
# ------------------------------------------------------------

with history_tab:

    st.subheader("Prediction History")

    if not st.session_state.prediction_history:

        st.info(
            "No tickets have been classified during "
            "this session."
        )

    else:
        history_df = pd.DataFrame(
            st.session_state.prediction_history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        history_csv = history_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Prediction History",
            data=history_csv,
            file_name="prediction_history.csv",
            mime="text/csv"
        )

        if st.button("Clear History"):
            st.session_state.prediction_history = []
            st.rerun()


# ------------------------------------------------------------
# MODEL INFORMATION TAB
# ------------------------------------------------------------

with model_tab:

    st.subheader("How the System Works")

    st.markdown(
        """
        1. The ticket subject and description are combined.
        2. Text is cleaned and normalized.
        3. TF-IDF converts the text into numerical features.
        4. Logistic Regression predicts the department.
        5. Prediction probabilities produce a confidence score.
        6. Low-confidence cases are sent for human review.
        7. Rule-based priority detection recommends an SLA.
        """
    )

    st.markdown("### Current model evaluation")

    evaluation_df = pd.DataFrame({
        "Metric": [
            "Test accuracy",
            "Cross-validation mean",
            "Training examples",
            "Number of categories"
        ],
        "Value": [
            "90.00%",
            "85.00%",
            "80",
            "4"
        ]
    })

    st.dataframe(
        evaluation_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Responsible-use note")

    st.write(
        "The system does not automatically assign tickets "
        "when model confidence is low. Those tickets are routed "
        "to a human-review queue, reducing the risk of incorrect "
        "department assignment."
    )

    st.markdown("### Future improvements")

    st.write(
        "The model can be improved with production ticket data, "
        "hyperparameter tuning, multilingual support, user feedback, "
        "model-drift monitoring and periodic retraining."
    )
