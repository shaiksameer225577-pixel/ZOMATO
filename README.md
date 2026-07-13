# 🍽️ Zomato Restaurant Recommendation System using SVD

## 📌 Project Overview

The **Zomato Restaurant Recommendation System** is a machine learning project that recommends restaurants to users using **Collaborative Filtering** with the **Singular Value Decomposition (SVD)** algorithm.

The system analyzes user–restaurant ratings to predict user preferences and recommends restaurants that users are most likely to enjoy. Since public Zomato datasets typically do not include user-specific ratings, a synthetic user-rating dataset is generated to simulate real-world recommendation scenarios.

---

# 🎯 Objectives

* Build a personalized restaurant recommendation system.
* Learn collaborative filtering using SVD.
* Perform data preprocessing and feature engineering.
* Evaluate recommendation quality using standard metrics.
* Develop a professional, GitHub-ready Data Science project.

---

# 🚀 Features

* Data preprocessing and cleaning
* Synthetic user-rating generation
* Collaborative Filtering using SVD
* Personalized Top-N restaurant recommendations
* Model evaluation using RMSE and MAE
* Save trained model for future predictions
* Export recommendations to CSV
* Modular and reusable project structure

---

# 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Surprise
* Scikit-Learn
* Matplotlib
* Seaborn
* Plotly
* Joblib
* Jupyter Notebook

---

# 📂 Project Structure

```text
Zomato-Recommendation-System/
│
├── data/
│   ├── zomato.csv
│   ├── restaurants_cleaned.csv
│   └── user_ratings.csv
│
├── models/
│   └── svd_model.pkl
│
├── outputs/
│   ├── top_10_recommendations.csv
│   └── evaluation_metrics.txt
│
├── src/
│   ├── data_preprocessing.py
│   ├── generate_user_ratings.py
│   ├── svd_model.py
│   ├── recommend.py
│   ├── evaluation.py
│   ├── utils.py
│   └── main.py
│
├── notebooks/
│   └── Zomato_SVD_Recommendation.ipynb
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

# 📊 Workflow

1. Load the Zomato restaurant dataset.
2. Clean and preprocess the data.
3. Generate synthetic user–restaurant ratings.
4. Train an SVD recommendation model.
5. Evaluate the model.
6. Generate Top-N personalized recommendations.
7. Save recommendations and evaluation results.

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Zomato-Recommendation-System.git
```

Navigate to the project folder:

```bash
cd Zomato-Recommendation-System
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Project

Navigate to the source folder:

```bash
cd src
```

Run the application:

```bash
python main.py
```

---

# 📈 Evaluation Metrics

The recommendation model is evaluated using:

* RMSE (Root Mean Square Error)
* MAE (Mean Absolute Error)
* Precision@K
* Recall@K
* F1 Score

---

# 📋 Sample Recommendation Output

| Restaurant        | Cuisine      | City      | Predicted Rating |
| ----------------- | ------------ | --------- | ---------------: |
| Barbeque Nation   | North Indian | Hyderabad |             4.92 |
| Paradise          | Biryani      | Hyderabad |             4.88 |
| Absolute Barbecue | BBQ          | Bengaluru |             4.84 |
| Cafe Coffee Day   | Cafe         | Chennai   |             4.81 |
| Domino's Pizza    | Fast Food    | Delhi     |             4.79 |

---

# 💼 Resume Description

Developed an end-to-end restaurant recommendation system using Singular Value Decomposition (SVD) and collaborative filtering. Performed data preprocessing, generated synthetic user–restaurant interactions, trained and evaluated the recommendation model, and produced personalized restaurant recommendations using Python and machine learning techniques.

---

# 🔮 Future Enhancements

* Streamlit web application
* Flask/FastAPI REST API
* User authentication
* Hybrid recommendation system
* Deep Learning recommendation models
* Real-time recommendation engine
* Cloud deployment
* Docker containerization

---

# 👨‍💻 Author

**Shaik Sameer**

* Data Analyst Aspirant
* Business Analyst Aspirant
* Data Science Enthusiast

---

# 📄 License

This project is licensed under the MIT License.

---

⭐ If you found this project useful, consider giving it a star on GitHub!
