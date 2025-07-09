# isPhish — Phishing Email Detection with Machine Learning

A lightweight web application that uses a machine learning model to classify email content as Safe or Phishing, with a confidence score and visual indicators to help users make informed decisions.

## Project Overview

Phishing remains one of the most common and dangerous attack vectors in cybersecurity. This application provides a fast, accessible tool for end users to assess suspicious emails by pasting the content into a web-based interface. It uses a Support Vector Machine (SVM) trained on thousands of labeled emails from public and synthetic datasets.

## Features

- SVM Classifier trained on multiple phishing datasets
- Classifies email content as `Safe` or `Phishing`
- Confidence score with a Plotly gauge chart
- Warning when confidence is low (near 50%)
- Visual diagnostics: PCA plot, confidence histogram, keyword analysis
- Built with Streamlit for easy browser-based access

## Project Structure

```
isPhish/
├── data/
│   ├── CEAS-08.csv
│   ├── TREC-06.csv
│   ├── TREC-07.csv
│   ├── Ling.csv
│   └── phishing_and_ham_emails.csv
├── isPhish.py          # Main Streamlit app
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## Installation

1. Clone the repository

```
git clone https://github.com/your-username/isPhish.git
cd isPhish
```

2. Install dependencies

```
pip install -r requirements.txt
```

3. Run the application

```
streamlit run isPhish.py
```

First run may take 10–20 minutes as the model is trained.

## Sample Emails to Test

**Safe:**
```
Hi team,
Just a reminder to review the updated Q3 timeline document I shared last Friday.
Thanks, Marcus
```

**Uncertain:**
```
Hi Daniel,
Your colleague, Anna White, has shared a document with you via our secure file sharing system.
```

**Phishing:**
```
There seems to be an issue with your payment. Please review the attached invoice to avoid service disruption.
```

## Machine Learning Details

- Algorithm: Support Vector Machine (Linear Kernel)
- Feature Extraction: TF-IDF (max 1000 features)
- Datasets:
  - CEAS-08, TREC-06, TREC-07, Ling
  - phishing_and_ham_emails.csv (synthetic via LLM)
- Train/Test Split: 67/33
- Libraries: scikit-learn, pandas, matplotlib, plotly, streamlit

## Visualizations

- PCA Decision Boundary: See how emails are classified in reduced space
- Confidence Histogram: Distribution of prediction probabilities
- Keyword Analysis: Most influential phishing/safe indicators

## License & Ethics

This project uses publicly available, anonymized datasets with no personal information. Synthetic data was generated for educational purposes. The project is open-source under the MIT License.

## Author

Devin Wilkes  
Email: wilkes.devinm@gmail.com  
GitHub: https://github.com/jet-blk
