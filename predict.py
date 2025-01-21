import joblib
import sys

def load_model():
    model = joblib.load('spam_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

def predict(text):
    model, vectorizer = load_model()
    text_transformed = vectorizer.transform([text])
    prediction = model.predict(text_transformed)
    return "This email is likely spam." if prediction[0] == 1 else "This email does not appear to be spam."

if __name__ == "__main__":
    text = sys.argv[1]
    prediction = predict(text)
    print (prediction)