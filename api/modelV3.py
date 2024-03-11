from flask import Flask, request, jsonify
import nltk
import string
from joblib import load
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re 
from cleantext import clean

app = Flask(__name__)


SuccessResponse = {"message": "success","status":200}


def remove_punctuation(sentence):
  """Removes punctuation characters from sentence, except for #."""
  return "".join([char for char in sentence if char not in string.punctuation or char == '#'])

def to_lowercase(sentence):
  """Converts sentence to lowercase."""
  return sentence.lower()

def stemming(sentence):
  """Reduces words to their root form using Porter Stemmer."""
  porter = nltk.stem.PorterStemmer()
  return " ".join([porter.stem(word) for word in sentence.split()])

def lemmatization(sentence):
  """Reduces words to their dictionary form using WordNet Lemmatizer."""
  wnl = nltk.WordNetLemmatizer()
  return " ".join([wnl.lemmatize(word) for word in sentence.split()])

def remove_stopwords(sentence):
  """Removes stop words """
  stop_words = stopwords.words('english')  # Download stopwords list (one-time)
  return " ".join([word for word in sentence.split() if word not in stop_words])

loaded_model = load('../trained_models/my_modelV3.joblib')
  
vectorizer = TfidfVectorizer(max_features=5000)

df = pd.read_csv("C:\\Users\\User\\Desktop\\output.csv")
df["sentence"] = df["sentence"].apply(remove_punctuation)
df["sentence"] = df["sentence"].apply(to_lowercase)
df["sentence"] = df["sentence"].apply(remove_stopwords)  
df["sentence"] = df["sentence"].apply(stemming)  
df["sentence"] = df["sentence"].apply(lemmatization)

X = df["sentence"]
X_features = vectorizer.fit_transform(X)


@app.route("/test", methods=["GET"])
def function():
    return jsonify(SuccessResponse)


@app.route("/predict", methods=["POST"])
def handle_request():
    data = request.form.get('sentence')

    if data:
        sentence = remove_punctuation(data.lower())
        sentence = clean(sentence, no_emoji=True)
        sentence = re.sub(r'\n', ' ', sentence)
        sentence = to_lowercase(sentence) 
        sentence = remove_stopwords(sentence) 
        sentence = stemming(sentence)
        sentence = lemmatization(sentence) 
        
        new_sentence = vectorizer.transform([sentence])
        prediction = loaded_model.predict(new_sentence)
        
        return jsonify({"message": "sentence predicted successfully", "data": prediction[0],"Proccessed":sentence,"status":200}), 200
    else:
        return jsonify({"error": "No data provided in request"}), 400



if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
