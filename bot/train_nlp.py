import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Training sentences
sentences = [
    "hello",
    "hi",
    "hey",
    "how can i improve focus",
    "how to concentrate better",
    "how to focus while studying",
    "tell me my focus score",
    "what is my focus score",
    "give study tips",
    "how to study better",
    "why is my focus low"
]

# Labels
labels = [
    "greeting",
    "greeting",
    "greeting",
    "improve_focus",
    "improve_focus",
    "improve_focus",
    "focus_score",
    "focus_score",
    "study_tips",
    "study_tips",
    "low_focus"
]

# TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sentences)

# Model
model = LogisticRegression()
model.fit(X, labels)

# Save model
joblib.dump(model, "model/nlp_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("✅ NLP model trained successfully")