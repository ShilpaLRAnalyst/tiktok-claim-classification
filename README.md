# 🎬 TikTok Video Claim Classification Web Application

An end-to-end Machine Learning and Natural Language Processing (NLP) pipeline built to classify TikTok video content into either a factual **Claim** or a personal **Opinion**. This project combines text feature extraction with social engagement metrics to serve live predictions via an interactive web interface.

🔗 **[Click Here to Open the Live Web Application](https://tiktok-claim-classification-hvqqvchfq9p48fqzetx4z8.streamlit.app/)**

---

## 🚀 Key Features & Pipeline Architecture
* **Unstructured Text Processing (NLP):** Utilizes a pre-trained `CountVectorizer` to tokenize, clean, and convert raw spoken video transcripts into high-dimensional numerical text matrices.
* **Feature Engineering Pipeline:** Dynamically scales and matches a 25-column alignment matrix combining live user-controlled engagement values (views, likes, shares, comments, downloads) with NLP features.
* **Predictive Performance:** Powered by a tuned **Random Forest Classifier** selected as the champion production model for its robust handling of complex, non-linear text and engagement interactions.
* **Cloud Architecture:** Packaged with standalone Python ecosystem dependencies and continuously deployed via Streamlit Community Cloud and GitHub version control workflows.

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Machine Learning & NLP:** Scikit-Learn, Pandas, NumPy, Pickle
* **Deployment & UI Framework:** Streamlit Framework, GitHub Webhooks, Cloud Infrastructure Server

---

## 📁 Repository Contents
* `TikTok_Claim_Classification_Project.ipynb`: The master engineering notebook containing exploratory data analysis, data cleaning, and feature encoding. It showcases the comparative evaluation of **Logistic Regression**, **XGBoost**, and **Random Forest** models, detailing hyperparameter tuning and champion model selection metrics.
* `app.py`: The master execution script hosting the user interface logic, database structural layout, and cloud prediction pipeline.
* `requirements.txt`: The isolated environment package log mapping exact third-party package dependencies required by the hosting web server.
* `tiktok_random_forest_backup.pkl`: The serialized binary file storing frozen random forest mathematical weights.
* `tiktok_vectorizer.pkl`: The serialized natural language vocabulary processing matrix mapping language features.

---

## 💡 Industry Application
Automated claim detection tools are critical components in modern social media ecosystems, helping content management teams scale moderation workflows, filter out misinformative claims, and segment opinion-based editorial content efficiently.
