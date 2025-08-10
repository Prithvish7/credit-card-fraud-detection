import streamlit as st
import pickle
import pandas as pd
import numpy as np
import io
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, classification_report

# Load models and test data for ROC curve
@st.cache_data(show_spinner=True)
def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

iso_forest = load_pickle("isolation_forest.pkl")
svm_clf = load_pickle("linear_svc.pkl")

# Load test data for performance tab
X_test = load_pickle("X_test.pkl")
y_test = load_pickle("y_test.pkl")
svm_scores = load_pickle("svm_scores.pkl")  # decision_function outputs for SVM on X_test
iso_scores = load_pickle("iso_scores.pkl")  # -decision_function outputs for Isolation Forest on X_test

# Scale function for Time and Amount (same scaler you used in training)
scaler = StandardScaler()
# Fit scaler on test set columns (or your training set if you have it)
scaler.fit(X_test[['Time', 'Amount']])

def scale_features(df_input):
    df_input = df_input.copy()
    df_input[['Time', 'Amount']] = scaler.transform(df_input[['Time', 'Amount']])
    return df_input

# Setup page config
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title & Navigation menu
st.markdown("""
<div style='display: flex; align-items: center; margin-bottom: 10px;'>
<h1 style="
    font-family: 'Poppins', sans-serif;
    font-size: 40px;
    background: linear-gradient(to right, #28a745, #007bff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-right: 20px;
">FraudShield AI</h1>
</div>
""", unsafe_allow_html=True)

selected = option_menu(
    menu_title=None,
    options=["Home", "About", "How it Works", "Performance", "Contact"],
    icons=["house", "info-circle", "gear", "bar-chart", "envelope"],
    menu_icon="cast",
    orientation="horizontal",
    default_index=0,
    styles={
        "container": {"padding": "0!important", "background-color": "#0ba334"},
        "icon": {"color": "white", "font-size": "16px"},
        "nav-link": {
            "font-size": "16px",
            "text-align": "center",
            "margin": "0px",
            "--hover-color": "#eee",
        },
        "nav-link-selected": {"background-color": "#1A083F", "color": "white"},
    }
)

if "history" not in st.session_state:
    st.session_state.history = []

if selected == "Home":
    st.markdown("""
    ### Upload transaction CSV or paste single transaction details to check for fraud
    
    **Required columns:**  
    - `Time` (seconds elapsed)  
    - `V1` to `V28` (PCA-transformed features, anonymized)  
    - `Amount` (transaction amount)  
    
    **Note:**  
    If you have raw transaction data (e.g., merchant info, card number, timestamps), this app currently expects preprocessed PCA features.  
    Future versions may support raw input with built-in preprocessing.
    """)

    # Sample CSV download
    sample_data = {
        'Time': [10000, 20000],
        **{f'V{i}': [0.1*i, -0.1*i] for i in range(1,29)},
        'Amount': [50.0, 120.0]
    }
    sample_df = pd.DataFrame(sample_data)

    csv_buffer = io.StringIO()
    sample_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download Sample CSV",
        data=csv_buffer.getvalue(),
        file_name="sample_transaction.csv",
        mime="text/csv",
        help="Download a sample CSV file with the correct columns and example values"
    )

    uploaded_file = st.file_uploader("Upload transaction sample CSV", type=["csv"])
    single_input = st.text_area(
        "Or paste single transaction as comma-separated values (Time,V1,V2,...,V28,Amount)",
        height=100,
        placeholder="e.g. 123456,-1.359807,-0.072781,...,149.62"
    )
    predict_btn = st.button("🚀 Predict Fraud")

    if predict_btn:
        try:
            if uploaded_file is not None:
                input_df = pd.read_csv(uploaded_file)
            elif single_input.strip():
                values = single_input.strip().split(",")
                if len(values) != 30:
                    st.error("Please enter exactly 30 values: Time + V1 to V28 + Amount")
                    st.stop()
                cols = ['Time'] + [f'V{i}' for i in range(1,29)] + ['Amount']
                input_df = pd.DataFrame([values], columns=cols)
                input_df = input_df.astype(float)
            else:
                st.warning("Please upload a CSV or paste transaction data")
                st.stop()

            input_df_scaled = scale_features(input_df)

            # Isolation Forest prediction
            iso_scores_pred = -iso_forest.decision_function(input_df_scaled)
            iso_pred = iso_forest.predict(input_df_scaled)
            iso_pred_binary = [1 if x == -1 else 0 for x in iso_pred]

            # LinearSVC prediction
            svm_scores_pred = svm_clf.decision_function(input_df_scaled)
            svm_pred_binary = [1 if s > 0 else 0 for s in svm_scores_pred]

            results = []
            for i in range(len(input_df)):
                results.append({
                    "Isolation Forest": "Fraud" if iso_pred_binary[i] else "Legit",
                    "IF Score": iso_scores_pred[i],
                    "Linear SVM": "Fraud" if svm_pred_binary[i] else "Legit",
                    "SVM Score": svm_scores_pred[i]
                })

                st.session_state.history.append({
                    "transaction": input_df.iloc[i].to_dict(),
                    "Isolation Forest": results[-1]["Isolation Forest"],
                    "IF Score": results[-1]["IF Score"],
                    "Linear SVM": results[-1]["Linear SVM"],
                    "SVM Score": results[-1]["SVM Score"]
                })

            for i, res in enumerate(results, 1):
                st.markdown(f"### Transaction #{i}")
                st.write(f"Isolation Forest Prediction: **{res['Isolation Forest']}** (Score: {res['IF Score']:.4f})")
                st.write(f"Linear SVM Prediction: **{res['Linear SVM']}** (Score: {res['SVM Score']:.4f})")
                st.markdown("---")

        except Exception as e:
            st.error(f"Error processing input: {e}")

    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 🕘 Prediction History (last 10)")
        for i, record in enumerate(reversed(st.session_state.history[-10:]), 1):
            st.markdown(f"**{i}.** Transaction: {record['transaction']}  \n"
                        f"IF: {record['Isolation Forest']} ({record['IF Score']:.4f}) | "
                        f"SVM: {record['Linear SVM']} ({record['SVM Score']:.4f})")

elif selected == "About":
    st.header("📌 About FraudShield AI")
    st.write("""
    **FraudShield AI** is a machine learning-powered app that detects fraudulent credit card transactions using Isolation Forest and Linear SVM models.
    It analyzes anonymized PCA-transformed features along with transaction time and amount.
    
    The models were trained on a highly imbalanced dataset and optimized for fraud detection.
    """)

elif selected == "How it Works":
    st.header("⚙️ How It Works")
    st.markdown("""
    - Upload a CSV file with transaction records or paste a single transaction's features.
    - The app scales the 'Time' and 'Amount' features.
    - It runs two models:
      - Isolation Forest (unsupervised anomaly detection)
      - Linear SVM (supervised classification)
    - Predictions and anomaly scores are displayed.
    - A prediction history keeps track of your past queries.
    """)

elif selected == "Performance":
    st.subheader("📈 Model Performance Overview")

    from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
    import plotly.graph_objects as go

    def plot_roc_curve(y_true, y_scores, model_name):
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        auc_score = roc_auc_score(y_true, y_scores)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{model_name} (AUC = {auc_score:.3f})'))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), showlegend=False))
        fig.update_layout(
            title=f'ROC Curve - {model_name}',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=700,
            height=500,
            font=dict(family="Segoe UI", size=14)
        )
        return fig

    # Convert predictions to binary for accuracy
    svm_pred_binary = [1 if s > 0 else 0 for s in svm_scores]
    iso_pred_binary = [1 if x == -1 else 0 for x in iso_forest.predict(X_test)]

    # Accuracy
    svm_accuracy = accuracy_score(y_test, svm_pred_binary) * 100
    iso_accuracy = accuracy_score(y_test, iso_pred_binary) * 100

    # ROC-AUC
    svm_roc_auc = roc_auc_score(y_test, svm_scores)
    iso_roc_auc = roc_auc_score(y_test, iso_scores)

    st.markdown("### Key Metrics")
    col1, col2 = st.columns(2)
    col1.metric("Linear SVM Accuracy", f"{svm_accuracy:.2f} %")
    col2.metric("Isolation Forest Accuracy", f"{iso_accuracy:.2f} %")

    col3, col4 = st.columns(2)
    col3.metric("Linear SVM ROC-AUC", f"{svm_roc_auc:.3f}")
    col4.metric("Isolation Forest ROC-AUC", f"{iso_roc_auc:.3f}")

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size:14px; color:#555;">
        ⚠️ <strong>Note:</strong> Due to the severe class imbalance in fraud detection datasets, accuracy can be misleadingly high.  
        ROC-AUC scores provide a better sense of the models’ ability to distinguish fraudulent transactions.
        </div>
        """, unsafe_allow_html=True
    )

    # Show ROC Curves
    st.plotly_chart(plot_roc_curve(y_test, svm_scores, "Linear SVM"))
    st.plotly_chart(plot_roc_curve(y_test, iso_scores, "Isolation Forest"))




elif selected == "Contact":
    st.header("📬 Contact Us")
    st.write("""
    **FraudShield AI Team**  
    📧 fraudshieldai@example.com  
    📱 +91-9876543210
    """)

# Footer
st.markdown(
    """
    <hr style="margin-top: 50px;">
    <p style='text-align: center; font-size: 14px; color: gray;'>
    Developed by Team <strong>FraudShield AI</strong><br>
    Last Updated: August 2025
    </p>
    """,
    unsafe_allow_html=True,
)
