# IMDB Movie Review Sentiment Analysis using Ensemble Learning

This project is a machine learning-based web application that classifies movie reviews as **Positive** or **Negative**. It uses an **Ensemble Learning** approach, combining multiple classifiers to achieve higher accuracy and robustness compared to individual models.

## 🚀 Features
* **Ensemble Model:** Combines Multinomial Naive Bayes, Logistic Regression, and SVM.
* **Text Preprocessing:** Robust cleaning using NLTK (Stopwords, Stemming).
* **Feature Extraction:** Implements TF-IDF Vectorization.
* **Web Interface:** Interactive UI built with Flask for real-time sentiment prediction.
* **EDA:** Includes detailed Exploratory Data Analysis with Word Clouds and performance metrics.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **ML Libraries:** Scikit-Learn, Pandas, Numpy, NLTK
* **Visualization:** Matplotlib, Seaborn, WordCloud
* **Web Framework:** Flask
* **Environment:** Jupyter Notebook (for training)

## 📋 Prerequisites
Ensure you have the following installed:
* Python 3.10+
* Git
* Pip (Python package manager)

## ⚙️ Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone [https://github.com/praveen-pydev/sentiment-analysis-imdb-movie-reviews.git](https://github.com/praveen-pydev/sentiment-analysis-imdb-movie-reviews.git)

   cd sentiment-analysis-imdb-movie-reviews

2. **Create a Virtual Environment**

   python -m venv .venv

3. **Activate the Environment**

    Windows: .venv\Scripts\activate

    Mac/Linux: source .venv/bin/activate

4. **Install Dependencies**

    python -m pip install --upgrade pip

    pip install -r requirements.txt

## 🖥️ Running the Application

1. **Setup the Kernel (Optional for Notebooks)**
If you want to run the training notebooks:

    pip install ipykernel

    python -m ipykernel install --user --name sentiment-analysis --display-name "Sentiment Analysis"

    then select the "Sentiment Analysis" kernel in Jupyter Notebook.

2. **Start the Flask Web App**
   
    cd app

    python app.py

Note: The app runs on http://127.0.0.1:5000 by default.

## 📊 Project Structure
* app.py: Flask application entry point.
* notebooks/: Contains the Jupyter notebook for model training and EDA.
* models/: Pre-trained models (Ensemble, SVM, NB, LR) and TF-IDF vectorizer.
* static/: CSS files and generated plots (Word Clouds, Evaluation charts).
* templates/: HTML files for the web interface.
* data/: (If included) The IMDB Dataset used for training.

## 👥 Contributors
1. Challagundla Praveen (Team Leader) - GitHub Profile
2. B. Kasi Rao
3. G. Sai Manikanta
4. J. Chiranjeevi
5. N. Dharma Sai

## 📜 License
This project is for educational purposes as part of the Final Year College Project at KITS Akshar Institute of Technology.