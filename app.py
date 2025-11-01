import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import random
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# Config
# ---------------------------
st.set_page_config(page_title="🏠 StayPals — Minimal + Matters + Chat", page_icon="🏠", layout="wide")

# ---------------------------
# Files
# ---------------------------
DATA_FILE = "users_data.csv"
CHATS_FILE = "chats.json"

# ---------------------------
# Features
# ---------------------------
FEATURES = [
    "Cleanliness", "Noise", "Smoking", "SleepTime",
    "StudyTime", "PetFriendly", "Cooking"
]
MATTERS_COLS = [f"Matters_{f}" for f in FEATURES]

# ---------------------------
# Ensure data files
# ---------------------------
def create_sample_users_csv(path):
    names = ["Manasvi","Aarav","Riya","Kabir","Ananya","Ishaan","Diya","Vivaan","Neha","Aditya"]
    genders = ["Female","Male"]
    rng = np.random.default_rng(42)
    rows = []
    for name in names:
        age = int(rng.integers(20, 30))
        gender = rng.choice(genders)
        row = {
            "Name": name, "Age": age, "Gender": gender,
            "Cleanliness": int(rng.integers(1,6)),
            "Noise": int(rng.integers(1,6)),
            "Smoking": int(rng.choice([0,1])),
            "SleepTime": int(rng.integers(1,6)),
            "StudyTime": int(rng.integers(1,6)),
            "PetFriendly": int(rng.choice([0,1])),
            "Cooking": int(rng.integers(1,6))
        }
        for mc in MATTERS_COLS:
            row[mc] = int(rng.choice([0,1]))
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)

def ensure_data():
    if not os.path.exists(DATA_FILE):
        create_sample_users_csv(DATA_FILE)
    if not os.path.exists(CHATS_FILE):
        with open(CHATS_FILE, "w") as f:
            json.dump([], f)

# ---------------------------
# Load / Save Helpers
# ---------------------------
def load_users():
    ensure_data()
    df = pd.read_csv(DATA_FILE)
    for c in FEATURES:
        if c not in df.columns:
            df[c] = 0
    for mc in MATTERS_COLS:
        if mc not in df.columns:
            df[mc] = 1
    return df

def save_users(df):
    df.to_csv(DATA_FILE, index=False)

# ---------------------------
# Recommendation Functions
# ---------------------------
def prepare_features_matrix(df):
    df = df.copy()
    for f in FEATURES:
        df[f] = df[f].fillna(0).astype(float)
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES])
    return pd.DataFrame(X, columns=FEATURES, index=df.index)

def get_user_matters(df, user_name):
    row = df[df["Name"] == user_name]
    if row.empty:
        return np.ones(len(FEATURES))
    vals = [int(row.iloc[0].get(f"Matters_{f}", 1)) for f in FEATURES]
    arr = np.array(vals, dtype=float)
    if arr.sum() == 0:
        arr = np.ones_like(arr)
    return arr / arr.sum()

def recommend(df, user_name, top_n=6):
    if user_name not in df["Name"].values:
        return pd.DataFrame()

    user_row = df[df["Name"] == user_name].iloc[0]
    user_gender = user_row["Gender"]
    df_same = df[df["Gender"] == user_gender].reset_index(drop=True)

    if len(df_same) < 2:
        return pd.DataFrame()

    X_df = prepare_features_matrix(df_same)
    matters = get_user_matters(df_same, user_name)
    X_weighted = X_df.values * matters[np.newaxis, :]
    sim = cosine_similarity(X_weighted)

    idx = df_same[df_same["Name"] == user_name].index[0]
    scores = sorted(enumerate(sim[idx]), key=lambda x: x[1], reverse=True)
    top_idx = [i for i, _ in scores if i != idx][:top_n]
    return df_same.iloc[top_idx].reset_index(drop=True)


# ---------------------------
# Fake Chat Window (Prototype)
# ---------------------------
FAKE_RESPONSES = [
    "Haha that's funny 😂",
    "Oh wow, tell me more!",
    "That's interesting 😄",
    "Same here 😅",
    "Lol you're cool 😎",
    "No way! Really?",
    "Agreed 💯"
]

def chat_page(me, partner):
    st.title(f"💬 Chat with {partner}")
    st.caption(f"You're chatting as **{me}**")

    # Each chat pair gets its own history
    chat_key = f"chat_{me}_{partner}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    chat_history = st.session_state[chat_key]

    # Show chat messages
    st.markdown("<hr>", unsafe_allow_html=True)
    for sender, msg in chat_history:
        align = "right" if sender == me else "left"
        bubble_color = "#DCF8C6" if sender == me else "#FFFFFF"
        st.markdown(
            f"""
            <div style="text-align:{align}; margin:8px;">
                <div style="
                    display:inline-block;
                    background-color:{bubble_color};
                    border-radius:10px;
                    padding:8px 12px;
                    max-width:70%;
                    color:#000;">
                    <b>{sender}:</b> {msg}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # User input
    st.markdown("<hr>", unsafe_allow_html=True)
    msg = st.text_input("Type your message", key=f"input_{chat_key}")

    col1, col2 = st.columns([1, 6])
    with col1:
        send = st.button("Send", key=f"send_{chat_key}")

    # Send message
    if send and msg.strip():
        chat_history.append((me, msg.strip()))
        chat_history.append((partner, random.choice(FAKE_RESPONSES)))
        st.session_state[chat_key] = chat_history
        st.rerun()



# ---------------------------
# MAIN UI
# ---------------------------
users = load_users()
query_params = st.query_params

# If chat opened in new tab
if "chat" in query_params:
    me = query_params.get("me", "You")
    partner = query_params.get("chat")
    chat_page(me, partner)
    st.stop()


# ---------------------------
# Main App
# ---------------------------
st.title("🏠 StayPals — Minimal Matches + Chat Sidebar")
st.caption("Create profile → pick what matters → get recommendations. Click 💬 to chat!")

# Sidebar
st.sidebar.header("Actions")
mode = st.sidebar.radio("Mode", ["Create Profile", "Get Recommendations"])

# Create Profile Mode
if mode == "Create Profile":
    st.subheader("📝 Create Profile")
    with st.form("profile_form"):
        name = st.text_input("Full name")
        age = st.number_input("Age", 18, 60, 22)
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        col1, col2 = st.columns(2)
        with col1:
            cleanliness = st.slider("Cleanliness", 1, 5, 3)
            noise = st.slider("Noise tolerance", 1, 5, 3)
            smoking = st.selectbox("Do you smoke?", ["No", "Yes"])
            sleeptime = st.slider("Sleep time", 1, 5, 3)
        with col2:
            studytime = st.slider("Study time", 1, 5, 3)
            pet = st.selectbox("Comfort with pets?", ["No", "Yes"])
            cooking = st.slider("Cooking habit", 1, 5, 3)

        st.markdown("### What *matters* to you?")
        select_all = st.checkbox("Select All", True)
        matters = {f: st.checkbox(f, select_all) for f in FEATURES}

        submit = st.form_submit_button("Save")

    if submit:
        if not name.strip():
            st.error("Name required.")
        else:
            row = {
                "Name": name, "Age": age, "Gender": gender,
                "Cleanliness": cleanliness, "Noise": noise,
                "Smoking": 1 if smoking == "Yes" else 0,
                "SleepTime": sleeptime, "StudyTime": studytime,
                "PetFriendly": 1 if pet == "Yes" else 0,
                "Cooking": cooking
            }
            for f in FEATURES:
                row[f"Matters_{f}"] = 1 if matters[f] else 0
            users = pd.concat([users, pd.DataFrame([row])], ignore_index=True)
            save_users(users)
            st.session_state["current_user"] = name
            st.success(f"Profile saved for {name} ✅")


# Get Recommendations Mode
if mode == "Get Recommendations":
    st.subheader("🔎 Find Matches")

    if users.empty:
        st.info("No users yet.")
    else:
        names = users["Name"].tolist()
        current = st.session_state.get("current_user", names[0])
        selected = st.selectbox("Select your profile", names, index=names.index(current))

        if st.button("Show recommendations"):
            recs = recommend(users, selected)

            if recs.empty:
                st.info("No matches found.")
            else:
                st.markdown("### 💡 Your Best Matches")
                cols_per_row = 3
                for i in range(0, len(recs), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for idx, col in enumerate(cols):
                        if i + idx < len(recs):
                            row = recs.iloc[i + idx]
                            with col:
                                st.markdown(
                                    f"""
                                    <div style="
                                        background-color:#1e1e1e;
                                        border-radius:18px;
                                        padding:20px;
                                        margin:10px;
                                        box-shadow:0 0 10px rgba(255,255,255,0.08);
                                        text-align:center;">
                                        <h3 style='color:#f0f0f0;margin-bottom:5px;'>{row['Name']}</h3>
                                        <p style='color:#ccc;margin:0;'>Age: {int(row['Age'])}</p>
                                        <p style='color:#aaa;margin:0;'>Gender: {row['Gender']}</p>
                                        <p style='color:#bbb;font-size:13px;'>🧹 Cleanliness: {row['Cleanliness']} | 🕓 Sleep: {row['SleepTime']}</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                                chat_link = f"?chat={row['Name']}&me={selected}"
                                st.markdown(
                                    f"<a href='{chat_link}' target='_blank'><button style='background-color:#4CAF50;color:white;border:none;padding:8px 16px;border-radius:8px;cursor:pointer;'>💬 Chat</button></a>",
                                    unsafe_allow_html=True,
                                )

st.markdown("---")
st.caption("Happy hunting 🏡")
