# 🏥 Axora: Clinical Document Classification System  

A production-grade **Machine Learning project** developed as part of **ITS 2130 – Machine Learning (Semester 4, 2025)**.  
This project was designed for **MedArchive Solutions**, a fictional healthcare technology company, to automate the classification and routing of clinical documents into their correct medical specialties.  

---

## 📖 Project Overview  

Healthcare providers often face a bottleneck in **manual triage of clinical transcriptions**, which is slow, error-prone, and costly. Misrouted documents can lead to **delays in treatment, billing issues, and operational inefficiencies**.  

Our solution:  
- Build an **end-to-end ML pipeline** that classifies clinical transcriptions into **13 medical specialties**.  
- Apply both **supervised learning (Softmax Regression)** and **unsupervised learning (KMeans clustering)**.  
- Deploy the trained model to **Google Cloud Vertex AI** for live predictions.  

Dataset used:  
👉 [hpe-ai/medical-cases-classification-tutorial](https://huggingface.co/datasets/hpe-ai/medical-cases-classification-tutorial) (2,460 anonymized clinical transcriptions).  

---

## 📂 Repository Structure  

```
/notebooks
 ├── 1_eda_and_preprocessing.ipynb
 ├── 2_classification_modeling.ipynb
 ├── 3_clustering_analysis.ipynb
 └── 4_model_deployment_and_testing.ipynb
/artifacts
 ├── model.joblib
 ├── tfidf_vectorizer.joblib
requirements.txt
README.md
```

---

## 📓 Jupyter Notebooks  

### 🔹 1. Exploratory Data Analysis & Preprocessing  
- Analyzes document lengths, distributions, and top keywords.  
- Visualizes **class imbalance** across 13 medical specialties.  
- Builds a **TF-IDF vectorizer** pipeline to convert raw text into numerical features.  
- Saves the fitted vectorizer for reuse in modeling.  

---

### 🔹 2. Classification Modeling & Evaluation  
- Trains a **Softmax Regression (Multinomial Logistic Regression)** classifier.  
- Uses **cross-validation** and basic hyperparameter tuning.  
- Evaluates performance with:  
  - ✅ Accuracy  
  - ✅ Precision, Recall, F1-score (per class)  
  - ✅ Confusion Matrix Heatmap  
- Packages the final **pipeline (vectorizer + classifier)** into `model.joblib`.  

---

### 🔹 3. Unsupervised Clustering Analysis  
- Applies **KMeans clustering** to TF-IDF vectors.  
- Uses **Elbow Method & Silhouette Score** to select the optimal number of clusters.  
- Extracts **top terms per cluster** and reviews sample documents for interpretation.  
- Visualizes clusters in 2D space using **TruncatedSVD**.  
- Provides **business insights**: identifying sub-specialties, triage patterns, and quality control opportunities.  

---

### 🔹 4. Deployment & Testing on Vertex AI  
- Uploads artifacts (`model.joblib`, `requirements.txt`) to **Google Cloud Storage**.  
- Registers the model in the **Vertex AI Model Registry**.  
- Deploys the model to a **live Vertex AI Endpoint**.  
- Sends real **clinical transcription test cases** to the endpoint and parses predictions.  
- Confirms successful cloud deployment.  

---

## 🚀 Deployment  

We deployed the trained pipeline to **Google Cloud Vertex AI**:  
1. Packaged `model.joblib` + `tfidf_vectorizer.joblib` + `requirements.txt`.  
2. Uploaded to **Google Cloud Storage**.  
3. Imported into **Vertex AI Model Registry**.  
4. Deployed a live **prediction endpoint**.  
5. Verified predictions with new unseen clinical cases.  

---

## 👥 Who Can Use This Project?  

- **Healthcare IT Teams** → Automate routing of clinical documents.  
- **Researchers** → Explore medical text mining, clustering, and classification.  
- **Students** → Learn supervised & unsupervised ML, and cloud deployment.  

---

## 🛠 How to Use  

1. Clone this repository:  
   ```bash
   git clone https://github.com/senumiminodya/axora-medarchive-solution.git
   cd clinical-document-classification
   ```
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Run notebooks in order:  
   - `1_eda_and_preprocessing.ipynb`  
   - `2_classification_modeling.ipynb`  
   - `3_clustering_analysis.ipynb`  
   - `4_model_deployment_and_testing.ipynb`  
4. For deployment: configure **Google Cloud SDK** and **Vertex AI** access before running Notebook 4.  

---

## 📌 Key Takeaways  

- Built a **robust ML pipeline** for medical text classification.  
- Balanced **academic rigor** (EDA, metrics, clustering) with **production readiness** (deployment).  
- Demonstrated the full **ML lifecycle**: from preprocessing → modeling → evaluation → clustering → deployment.  

---

## 📜 License  

This project is developed as part of **ITS 2130: Machine Learning – Group Project (Semester 4, 2025)**.  
It is intended for **educational and research purposes only**.  

---

✨ With this project, MedArchive Solutions can **automatically classify clinical documents**, reducing human error, saving time, and improving patient care efficiency.  
