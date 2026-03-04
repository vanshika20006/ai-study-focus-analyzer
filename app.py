import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
from bot.chatbot import study_bot

# -----------------------------
# Session State
# -----------------------------
if "focus_score" not in st.session_state:
    st.session_state.focus_score = 0

if "show_bot" not in st.session_state:
    st.session_state.show_bot = False

# -----------------------------
# Load ML Model
# -----------------------------
model = joblib.load("model/focus_model.pkl")

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="AI Study Focus Analyzer",
    page_icon="🧠",
    layout="centered"
)

# -----------------------------
# Floating Button Style
# -----------------------------
st.markdown("""
<style>
div.stButton > button:first-child {
position: fixed;
bottom: 25px;
right: 25px;
background-color: #ff4b4b;
color: white;
border-radius: 50%;
height: 65px;
width: 65px;
font-size: 28px;
box-shadow: 0px 4px 15px rgba(0,0,0,0.4);
z-index: 9999;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------
st.title("🧠 AI Study Focus Analyzer")
st.write("Fully Offline | ML Powered | Study Smart 📚")

st.divider()

# -----------------------------
# User Inputs
# -----------------------------
study_time = st.slider("📖 Study Time (minutes)", 0, 300, 60)
phone_time = st.slider("📱 Phone Usage (minutes)", 0, 300, 30)
breaks = st.slider("☕ Breaks Taken", 0, 10, 2)
mood = st.slider("🙂 Mood Level (1 = Bad, 5 = Great)", 1, 5, 3)

st.divider()

# -----------------------------
# Focus Prediction
# -----------------------------
if st.button("🔍 Analyze My Focus"):

    input_data = np.array([[study_time, phone_time, breaks, mood]])
    focus_score = int(model.predict(input_data)[0])

    st.session_state.focus_score = focus_score

    st.subheader("🎯 Your Focus Score")

    st.progress(min(focus_score, 100))
    st.success(f"Your Focus Score is **{focus_score}/100**")

    if focus_score >= 80:
        st.balloons()
        st.write("🔥 Excellent focus! Keep it up.")

    elif focus_score >= 50:
        st.warning("🙂 Decent focus, but you can improve.")

    else:
        st.error("⚠️ Low focus detected. Let's fix it!")

st.divider()

# -----------------------------
# Study Analytics Dashboard
# -----------------------------
st.subheader("📊 Study Analytics Dashboard")

col1, col2 = st.columns(2)

# Bar Chart
with col1:
    fig, ax = plt.subplots()

    labels = ["Study Time", "Phone Usage", "Breaks"]
    values = [study_time, phone_time, breaks]

    ax.bar(labels, values)
    ax.set_title("Study Behaviour Analysis")
    ax.set_ylabel("Minutes / Count")

    st.pyplot(fig)

# Pie Chart
with col2:
    fig2, ax2 = plt.subplots()

    sizes = [study_time, phone_time]
    labels = ["Study Time", "Phone Usage"]

    ax2.pie(sizes, labels=labels, autopct='%1.1f%%')
    ax2.set_title("Focus vs Distraction")

    st.pyplot(fig2)

st.divider()

# -----------------------------
# Floating Chatbot Button
# -----------------------------
if st.button("🤖"):
    st.session_state.show_bot = not st.session_state.show_bot

# -----------------------------
# Chatbot Sidebar
# -----------------------------
if st.session_state.show_bot:

    with st.sidebar:

        st.title("🤖 AI Study Assistant")

        user_message = st.text_input("Ask me anything about your study focus:")

        if st.button("Send"):

            if user_message.strip() == "":
                st.warning("Please enter a message first.")

            else:
                reply = study_bot(user_message, st.session_state.focus_score)
                st.success(reply)

        st.markdown("""
💡 **Try asking**

• Hey  
• Hello  
• How can I improve focus?  
• Why is my focus low?  
• Tell me my focus score  
• Give study tips
""")