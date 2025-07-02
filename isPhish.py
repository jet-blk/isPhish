import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import plotly.graph_objects as go

# Load data
@st.cache_data
def load_and_prepare_data():
    df1 = pd.read_csv("data/phishing_and_ham_emails.csv")
    df2 = pd.read_csv("data/CEAS-08.csv")
    df3 = pd.read_csv("data/Ling.csv")
    df4 = pd.read_csv("data/TREC-07.csv")

    for df in [df1, df2, df3, df4]:
        df['Text'] = df['subject'].fillna('') + " " + df['body'].fillna('')
        df.dropna(subset=['Text'], inplace=True)

    combined_df = pd.concat([df1[['Text', 'label']], df2[['Text', 'label']], df3[['Text', 'label']], df4[['Text', 'label']]], ignore_index=True)
    return combined_df

data = load_and_prepare_data()

# Train model
@st.cache_resource
def train_model(df):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(df['Text'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    svm = SVC(kernel='linear', probability=True, random_state=42)
    svm.fit(X_train, y_train)

    return svm, vectorizer, X, y, X_train, X_test, y_train, y_test

svm, vectorizer, X, y, X_train, X_test, y_train, y_test = train_model(data)

# Streamlit application
st.title("Phishing Email Detector")
st.markdown("Paste plaintext email content into the box and click **Classify Email** to determine if it's safe or not.")

user_email = st.text_area("Email Content", height=200)

if st.button("Classify Email"):
    if not user_email.strip():
        st.warning("Please insert email content.")
    else:
        X_input = vectorizer.transform([user_email])
        prediction = svm.predict(X_input)[0]
        confidence = svm.predict_proba(X_input)[0][1]

        label = "🚨 Phishing" if prediction == 1 else "✅ Safe"
        if 0.4 <= confidence <= 0.6:
                label = "⚠️ Uncertain"
                st.warning("⚠️ Low confidence prediction: The model's classification is uncertain as the email has characteristics of both safe and phishing emails. Please review manually.")
        st.subheader(f"**Prediction**: {label}")
        st.write("### Confidence Gauge")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            number={'suffix': " prob"},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.4], 'color': "lightgreen"},
                    {'range': [0.4, 0.5], 'color': "yellow"},
                    {'range': [0.5, 0.6], 'color': "orange"},
                    {'range': [0.6, 1.0], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': confidence
                }
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### Top phishing indicators (from training)")
        feature_names = vectorizer.get_feature_names_out()
        coefs = svm.coef_.toarray().flatten()
        top_phish = np.argsort(coefs)[-10:]
        for i in reversed(top_phish):
            st.write(f"- **{feature_names[i]}**: {coefs[i]:.3f}")

# Metrics
with st.expander("Model Evaluation Metrics"):
    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"**Test Accuracy:** `{acc:.2f}`")
    st.text(classification_report(y_test, y_pred))

# Histogram
with st.expander("Phishing Probability Distribution (Test Set)"):
    probs = svm.predict_proba(X_test)
    fig, ax = plt.subplots()
    sns.histplot(probs[:, 1], bins=20, kde=True, ax=ax)
    ax.set_title("Distribution of Phishing Probability Scores")
    ax.set_xlabel("Phishing Probability")
    ax.set_ylabel("Frequency")
    ax.grid(True)
    st.pyplot(fig)

# PCA plot
with st.expander("Visual Decision Boundary (PCA-Reduced)"):
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X.toarray())

    svm_pca = SVC(kernel='linear')
    svm_pca.fit(X_pca, y)

    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    mesh = np.c_[xx.ravel(), yy.ravel()]
    Z = svm_pca.decision_function(mesh).reshape(xx.shape)

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.contourf(xx, yy, Z > 0, alpha=0.2, cmap=ListedColormap(['green', 'red']))
    ax2.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
    ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=ListedColormap(['green', 'red']), edgecolors='k', alpha=0.7)
    ax2.set_title("SVM Decision Boundary in PCA-Reduced Feature Space")
    ax2.set_xlabel("PCA Component 1")
    ax2.set_ylabel("PCA Component 2")
    ax2.grid(True)
    st.pyplot(fig2)