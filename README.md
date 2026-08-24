# Smartphone Price Prediction 📱💰

> **Predict smartphone prices intelligently using Machine Learning**

[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white)](https://smartphone-price-prediction-rohan.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/rohanjpatil/Smartphone-Price-Prediction)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](https://opensource.org/licenses/MIT)

---

## 📌 Overview

**Smartphone Price Prediction** is a machine learning-powered web application that estimates the price of a smartphone based on its technical specifications. By simply entering key features like RAM, storage, camera quality, battery capacity, and more, users can get an instant price prediction.

This project demonstrates the complete **end-to-end machine learning lifecycle** — from data exploration and model training to deployment as an interactive web application.

---

## 🚀 Live Demo

**Try it out here:** [https://smartphone-price-prediction-rohan.streamlit.app/](https://smartphone-price-prediction-rohan.streamlit.app/)

The live application is hosted on **Streamlit Community Cloud**, making it accessible to anyone with an internet connection.

---

## ✨ Features

- 🔮 **Real-time Price Prediction:**  Get instant price estimates based on smartphone specifications
- 📊 **Interactive UI:**  Clean, user-friendly interface built with Streamlit
- 🧠 **Machine Learning Powered:**  Leverages trained regression models for accurate predictions
- 📱 **Multiple Specifications:**  Input features like RAM, ROM, camera resolution, battery, and more
- ⚡ **Fast & Responsive:**  Lightweight application with quick inference

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit |
| **ML Framework** | Scikit-learn |
| **Language** | Python 3.8+ |
| **Model Serialization** | Pickle |
| **Development** | Jupyter Notebook |
| **Deployment** | Streamlit Community Cloud |

---

## 📁 Project Structure

```
Smartphone-Price-Prediction/
├── app.py                          # Streamlit web application
├── Smarphones.ipynb                # Jupyter Notebook with EDA & model training
├── smartphone_price_model.pkl      # Trained machine learning model
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## 🏗️ How It Works

1. **Data Collection & Exploration:**  The dataset containing smartphone specifications and their corresponding prices is analyzed and preprocessed.

2. **Model Training:**  Multiple machine learning algorithms are evaluated, and the best-performing model is selected and saved.

3. **Web Application:**  Streamlit provides an intuitive interface where users can input specifications and receive instant predictions.

4. **Deployment:**  The application is deployed on Streamlit Cloud for public access.

---

## 🔧 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/rohanjpatil/Smartphone-Price-Prediction.git
   cd Smartphone-Price-Prediction
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application locally**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

---

## 📊 Model Performance

The project explores various supervised learning techniques for price prediction, including:

- **Linear Regression**
- **Random Forest Regressor**
- **Support Vector Regression (SVR)**
- **K-Nearest Neighbors (KNN)**
- **Logistic Regression**

The best-performing model is serialized as `smartphone_price_model.pkl` and used in the production application.

---


## 🤝 Contributing

Contributions are welcome! If you'd like to improve this project:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the **MIT License** see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Dataset providers and open-source contributors
- The Streamlit team for their amazing framework
- Scikit-learn community for robust ML tools

---

## 📬 Contact

**Rohan J. Patil**

[![GitHub](https://img.shields.io/badge/GitHub-rohanjpatil-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/rohanjpatil)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Rohan%20Patil-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rohan-j-patil/)

---

⭐ **If you found this project useful, please consider giving it a star on GitHub!** ⭐
