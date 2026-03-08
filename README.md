# 📦 Demand Forecasting for Supply Chain using Machine Learning

## 📌 Project Overview

Demand forecasting is a critical task in supply chain management.
This project predicts product demand using **Machine Learning (LightGBM)** based on historical data and engineered time features.

The model analyzes patterns in the data to forecast future demand, helping businesses make better decisions in **inventory management, production planning, and logistics**.


## 🚀 Features

* 📊 Demand prediction using **LightGBM Regressor**
* 🧠 Feature engineering with **cyclical time features**
* ⚡ Fast data loading using **Feather format**
* 📉 Visualization of demand trends
* 🌐 Interactive **Streamlit Web App**
* 💾 Saved ML model using **Joblib**



## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **LightGBM**
* **Matplotlib**
* **Seaborn**
* **Streamlit**
* **Joblib**



## 📂 Project Structure


Demand-Forecasting
│
├── app.py
├── LightGBM_model.joblib
├── scaler.joblib
│
├── dataset
│   └── dimred_df.feather
│
├── requirements.txt
└── README.md


## ⚙️ Installation

Clone the repository:


git clone https://github.com/your-username/demand-forecasting-ml.git
cd demand-forecasting-ml

Install dependencies:


pip install -r requirements.txt


## ▶️ Run the Streamlit App


streamlit run app.py


Then open in your browser:


http://localhost:8501


## 📊 Model Details

Algorithm used:

* **LightGBM Regressor**

Features used for prediction:

* `date_id`
* `PC1`
* `PC2`
* `day_sin`
* `day_cos`
* `month_sin`
* `month_cos`

Evaluation Metrics:

* **RMSE (Root Mean Squared Error)**
* **MAE (Mean Absolute Error)**



## 📈 Sample Output

The model predicts the **expected demand value** based on the provided input features.

The Streamlit app allows users to:

* Enter feature values
* Predict demand instantly
* View historical demand trends



## 🎯 Future Improvements

* Add **deep learning models (LSTM / Transformer)**
* Add **multiple product forecasting**
* Deploy the app on **Streamlit Cloud**
* Add **interactive dashboards**


## 👩‍💻 Author

**Thota Anushka**
BTech Student | Machine Learning Enthusiast



## ⭐ Support

If you found this project useful, please consider **starring ⭐ the repository**.
