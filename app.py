import streamlit as st
import pandas as pd
import numpy as np
import os
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="StayPals", layout="wide")

DATA_FILE = "users_data.csv"
FEATURES = ["Cleanliness","Noise","Smoking","SleepTime","StudyTime","PetFriendly","Cooking"]
MATTERS = [f"Matters_{f}" for f in FEATURES]

# -----------------------------
# Ensure data file
# -----------------------------
def create_sample_csv():
    names = ["Aarav","Diya","Ishaan","Neha","Aditya","Riya","Kabir"]
    genders = ["Male","Female"]
    data = []
    for n in names:
        g = random.choice(genders)
        entry = {
            "Name": n,
            "Age": random.randint(18,30),
            "Gender": g,
        }
        for f in FEATURES:
            entry[f] = random.randint(1,5)
        for m in MATTERS:
            entry[m] = random.choice([0,1])
        data.append(entry)
    pd.DataFrame(data).to_csv(DATA_FILE,index=False)

if not os.path.exists(DATA_FILE):
    create_sample_csv()

# -----------------------------
# Data helpers
# -----------------------------
def load_users():
    df = pd.read_csv(DATA_FILE)
    return df

def save_users(df):
    df.to_csv(DATA_FILE,index=False)

def prepare_features_matrix(df):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES])
    return pd.DataFrame(X, columns=FEATURES)

def get_user_matters(df, name):
    row = df[df["Name"]==name]
    if row.empty:
        return np.ones(len(FEATURES))
    arr = np.array([int(row.iloc[0][f"Matters_{f}"]) for f in FEATURES])
    if arr.sum()==0:
        arr = np.ones_like(arr)
    return arr/arr.sum()

def recommend(df, name):
    if name not in df["Name"].values:
        return pd.DataFrame()
    user = df[df["Name"]==name].iloc[0]
    gender = user["Gender"]
    same = df[df["Gender"]==gender].reset_index(drop=True)
    if len(same)<2: return pd.DataFrame()
    X = prepare_features_matrix(same)
    matters = get_user_matters(same,name)
    Xw = X.values * matters
    sim = cosine_similarity(Xw)
    idx = same[same["Name"]==name].index[0]
    scores = sorted(enumerate(sim[idx]), key=lambda x:x[1], reverse=True)
    top = [i for i,_ in scores if i!=idx][:6]
    return same.iloc[top]

# -----------------------------
# UI layout
# -----------------------------
st.sidebar.title("StayPals")
mode = st.sidebar.radio("Navigate",["Create Profile","Get Recommendations"])

users = load_users()

if mode=="Create Profile":
    st.header("📝 Create Profile")
    with st.form("profile"):
        name = st.text_input("Full name")
        age = st.number_input("Age",18,60,22)
        gender = st.selectbox("Gender",["Male","Female","Other"])
        cleanliness = st.slider("Cleanliness",1,5,3)
        noise = st.slider("Noise tolerance",1,5,3)
        smoking = st.selectbox("Do you smoke?",["No","Yes"])
        sleeptime = st.slider("Sleep time",1,5,3)
        studytime = st.slider("Study time",1,5,3)
        pet = st.selectbox("Comfort with pets?",["No","Yes"])
        cooking = st.slider("Cooking habit",1,5,3)
        st.write("### What matters most?")
        select_all = st.checkbox("Select All",True)
        matters = {f:st.checkbox(f,select_all) for f in FEATURES}
        submit = st.form_submit_button("Save")
    if submit:
        row = {
            "Name":name,"Age":age,"Gender":gender,
            "Cleanliness":cleanliness,"Noise":noise,
            "Smoking":1 if smoking=="Yes" else 0,
            "SleepTime":sleeptime,"StudyTime":studytime,
            "PetFriendly":1 if pet=="Yes" else 0,"Cooking":cooking
        }
        for f in FEATURES:
            row[f"Matters_{f}"] = 1 if matters[f] else 0
        users = pd.concat([users,pd.DataFrame([row])],ignore_index=True)
        save_users(users)
        st.session_state["current_user"] = name
        st.success(f"Profile saved for {name} ✅")

elif mode=="Get Recommendations":
    st.header("🔍 Find Matches")
    if users.empty:
        st.info("No users yet.")
    else:
        current = st.session_state.get("current_user",users["Name"].iloc[0])
        sel = st.selectbox("Select your profile",users["Name"].tolist(),index=users["Name"].tolist().index(current))
        if st.button("Show recommendations"):
            recs = recommend(users,sel)
            if recs.empty:
                st.info("No matches found.")
            else:
                st.session_state["recs"] = recs
                st.session_state["selected_user"] = sel

# Show recommendations if available
if "recs" in st.session_state:
    recs = st.session_state["recs"]
    st.subheader("💡 Recommended Matches")
    cols = st.columns(3)
    for i,(_,r) in enumerate(recs.iterrows()):
        with cols[i%3]:
            st.markdown(f"""
                <div style="background:#f8f9fa;border-radius:14px;padding:15px;
                            margin:8px;box-shadow:0 0 6px rgba(0,0,0,0.08);text-align:center;">
                    <h4>{r['Name']}</h4>
                    <p>Age: {r['Age']} | {r['Gender']}</p>
                </div>
            """,unsafe_allow_html=True)
            if st.button(f"💬 Chat with {r['Name']}",key=f"chat_{r['Name']}"):
                st.session_state["chat_partner"] = r["Name"]

# -----------------------------
# Chat popup prototype
# -----------------------------
if "chat_partner" in st.session_state:
    partner = st.session_state["chat_partner"]

    st.markdown("""
        <style>
        .chat-popup {
            position: fixed;
            bottom: 30px;
            right: 30px;
            width: 320px;
            background-color: #ffffff;
            border: 1px solid #ccc;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 9999;
            padding: 12px;
            font-family: 'Segoe UI', sans-serif;
        }
        .chat-header {
            font-weight: 600;
            font-size: 16px;
            margin-bottom: 8px;
            color: #333;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .chat-box {
            height: 240px;
            overflow-y: auto;
            padding: 8px;
            background-color: #f8f9fa;
            border-radius: 6px;
        }
        .chat-msg {
            margin: 6px 0;
            padding: 6px 10px;
            border-radius: 8px;
            display: inline-block;
            max-width: 80%;
            line-height: 1.3;
        }
        .user {
            background-color: #2563eb;
            color: white;
            align-self: flex-end;
            text-align: right;
        }
        .bot {
            background-color: #e9ecef;
            color: #333;
            align-self: flex-start;
            text-align: left;
        }
        </style>
    """, unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"sender": "bot", "text": f"Hey, I'm {partner} 👋"},
            {"sender": "bot", "text": "How’s your day going?"}
        ]

    # Display chat
    chat_html = f"""
    <div class="chat-popup">
        <div class="chat-header">
            Chat with {partner}
            <span style="cursor:pointer;color:#888;" onclick="window.parent.postMessage('closeChat','*')">✖</span>
        </div>
        <div class="chat-box" id="chatbox">
    """
    for msg in st.session_state.chat_history:
        role = "user" if msg["sender"]=="user" else "bot"
        chat_html += f"<div class='chat-msg {role}'>{msg['text']}</div><br>"
    chat_html += "</div></div>"
    st.markdown(chat_html,unsafe_allow_html=True)

    msg = st.text_input("Type a message", key="chat_input")
    if st.button("Send", key="send_btn"):
        if msg.strip():
            st.session_state.chat_history.append({"sender":"user","text":msg})
            reply = random.choice(["Haha, same here 😄","That's interesting!","Tell me more!","Nice!","Cool 😎","Oh wow, really?"])
            st.session_state.chat_history.append({"sender":"bot","text":reply})
            st.rerun()

    # JS close button support
    st.markdown("""
    <script>
    window.addEventListener("message", (event) => {
        if(event.data === "closeChat"){
            window.location.reload();
        }
    });
    </script>
    """, unsafe_allow_html=True)
