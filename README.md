# Spam Email Detector

A machine learning application to detect spam emails using the **Scikit-learn** library. This project includes tools to train the model, make predictions, and containerize the application with Docker for easy deployment.

---

## **Features**
- **Spam Detection Model**:
  - Utilizes a **Multinomial Naive Bayes** classifier.
  - Converts email text into a **TF-IDF** matrix for feature extraction.
- **Prediction Script**:
  - Accepts an email text as input and predicts whether it is spam.
  - Saves the trained model and vectorizer for reuse using `joblib`.
- **Docker Support**:
  - Containerizes the application for deployment using a lightweight Python 3.12 Slim image.

---

## **Project Structure**
```
.
├── spamdetector.py   # Script to train the spam detection model
├── predict.py        # Script to make predictions using the trained model
├── Dockerfile        # Docker configuration file
```

---

## **Getting Started**

### **Clone the Repository**
```bash
git clone https://github.com/yourusername/spam-email-detector.git
cd spam-email-detector
```

### **Install Dependencies**
```bash
pip install pandas scikit-learn joblib
```

### **Train the Model**
Run the `spamdetector.py` script to train the model:
```bash
python3 spamdetector.py
```

This will:
- Train the model on a labeled dataset.
- Save the trained model as `spam_model.pkl` and the vectorizer as `vectorizer.pkl`.
- Display the model's accuracy and classification report.

### **Make a Prediction**
Use the `predict.py` script to classify an email as spam or not:
```bash
python3 predict.py "Your email text here"
```

The script will output whether the email is likely spam or not.

---

## **Run in Docker**

1. **Build the Docker Image**:
   ```bash
   docker build -t spam-email-detector .
   ```

2. **Run the Docker Container**:
   ```bash
   docker run --rm spam-email-detector
   ```

The container will execute `spamdetector.py` by default.

---

## **Output**
- **Training**:
  - Saves `spam_model.pkl` (trained model) and `vectorizer.pkl` (TF-IDF vectorizer).
  - Outputs:
    - Model accuracy.
    - Classification report with precision, recall, and F1 scores.
- **Prediction**:
  - Outputs whether the email text is spam or not.

---

## **Technologies Used**
- **Python**: Programming language.
- **Scikit-learn**: Machine learning library.
- **Pandas**: Data manipulation and analysis.
- **Joblib**: Model persistence.
- **Docker**: Containerization.

---

## **Why This Project?**
This project demonstrates:
- Practical application of machine learning for spam detection.
- Text preprocessing and feature extraction using TF-IDF.
- Deployment scalability using Docker.

Perfect for learning or deploying a lightweight spam detection system!

---

## **Contributing**
Feel free to fork the repository and submit pull requests. Suggestions and improvements are always welcome!