import joblib
import numpy as np

# Load NLP model
model = joblib.load("model/nlp_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")


def study_bot(user_input, focus_score):

    # Convert text to vector
    X = vectorizer.transform([user_input])

    # Predict probabilities
    probs = model.predict_proba(X)[0]
    intent = model.classes_[np.argmax(probs)]
    confidence = np.max(probs)

    # Fallback if confidence low
    if confidence < 0.40:
        return (
            "🤖 I'm not fully sure what you mean.\n\n"
            "Try asking about:\n"
            "• improving focus\n"
            "• study tips\n"
            "• your focus score"
        )

    # Greeting
    if intent == "greeting":
        return "👋 Hello! I'm your AI Study Assistant. How can I help you?"

    # Improve focus
    elif intent == "improve_focus":
        return (
            "📈 Tips to improve focus:\n"
            "- Reduce phone usage 📵\n"
            "- Study in 45–50 minute slots ⏱️\n"
            "- Take short breaks ☕\n"
            "- Keep your study space clean 🧹"
        )

    # Focus score
    elif intent == "focus_score":
        return f"🎯 Your focus score is **{focus_score}/100**."

    # Study tips
    elif intent == "study_tips":
        return (
            "📚 Study Tips:\n"
            "- Set daily study goals 🎯\n"
            "- Avoid distractions 📵\n"
            "- Revise before sleeping 😴"
        )

    # Low focus
    elif intent == "low_focus":
        return (
            "⚠️ Low focus may be caused by distractions.\n"
            "Try meditation, exercise, or short walks."
        )

    else:
        return "🤖 I'm still learning. Try asking about study or focus."