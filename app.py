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
def get_fake_response(user_msg):
    msg = user_msg.lower().strip()

    greetings = ["hi", "hello", "hey", "heyy", "heyyy", "yo"]
    goodbyes = ["bye", "goodbye", "see ya", "see you", "later"]
    thanks = ["thank", "thanks", "thank you", "appreciate"]
    questions = ["how are you", "what's up", "how’s it going", "how r u"]

    if any(g in msg for g in greetings):
        return random.choice([
            "Heyy 👋 how’s it going?",
            "Hello there 😄",
            "Hey! How are you?",
            "Hi hi! 👋",
            "Yo! What’s up?"
        ])
    elif any(q in msg for q in questions):
        return random.choice([
            "I’m good, just chilling 😌 you?",
            "Doing great! How about you?",
            "All good here 😄 what about you?",
            "Pretty relaxed tbh 😎 you?"
        ])
    elif any(t in msg for t in thanks):
        return random.choice([
            "Aww you’re welcome 😊",
            "Anytime!",
            "Glad to help 😄",
            "No problem at all 🙌"
        ])
    elif any(b in msg for b in goodbyes):
        return random.choice([
            "Bye bye 👋 take care!",
            "See you soon 😄",
            "Later! ✌️",
            "Goodbye 👋"
        ])
    elif "?" in msg:
        return random.choice([
            "Hmm good question 🤔",
            "That’s interesting, what do you think?",
            "Not sure honestly 😅",
            "Haha maybe! 😄"
        ])
    else:
        return random.choice([
            "Haha that’s cool 😄",
            "Nicee 😎",
            "Oh really? Tell me more!",
            "I get that 😌",
            "Lol same 😂",
            "That’s awesome!",
            "You sound fun haha",
            "Totally agree 😁",
            "Omg same here 😅",
            "That’s kinda nice tbh 😌"
        ])

def chat_page(me, partner):
    st.title(f"💬 Chat with {partner}")
    st.caption(f"You're chatting as **{me}**")

    chat_key = f"chat_{me}_{partner}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    chat_history = st.session_state[chat_key]

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

    st.markdown("<hr>", unsafe_allow_html=True)
    msg = st.text_input("Type your message", key=f"input_{chat_key}")

    col1, col2 = st.columns([1, 6])
    with col1:
        send = st.button("Send", key=f"send_{chat_key}")

    if send and msg.strip():
        user_msg = msg.strip()
        chat_history.append((me, user_msg))
        bot_reply = get_fake_response(user_msg)
        chat_history.append((partner, bot_reply))
        st.session_state[chat_key] = chat_history
        st.rerun()

# ---------------------------
# MAIN UI
# ---------------------------
users = load_users()
query_params = st.query_params

if "chat" in query_params:
    me = query_params.get("me", "You")
    partner = query_params.get("chat")
    chat_page(me, partner)
    st.stop()

st.title("🏠 StayPals")
st.caption("Create profile → pick what matters → get recommendations. Click 💬 to chat!")

st.sidebar.header("Actions")
mode = st.sidebar.radio("Mode", ["Create Profile", "Get Recommendations"])

# ---------------------------
# 📂 Data Tools Section (for admins / debugging)
# ---------------------------
st.sidebar.header("📂 Data Tools")

if st.sidebar.button("View Current Users Data"):
    df = load_users()
    st.write("### 👥 Current Users Data")
    st.dataframe(df)

csv_data = load_users().to_csv(index=False).encode("utf-8")
st.sidebar.download_button(
    label="⬇️ Download users_data.csv",
    data=csv_data,
    file_name="users_data.csv",
    mime="text/csv"
)

# ---------------------------
# ---------------------------
# ---------------------------
# Create Profile Mode
# ---------------------------
if mode == "Create Profile":
    st.subheader("📝 Create Profile")

    # --- BASIC PROFILE FORM ---
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

        st.markdown("### 💬 Choose tags that describe you best:")
        TAG_OPTIONS = [
            "Chill", "Extrovert", "Introvert", "Organized", "Spontaneous",
            "Homebody", "Party-lover", "Minimalist", "Night owl", "Early bird",
            "Gym freak", "Easygoing", "Creative", "Adventurous", "Talkative", "Calm",
            "Netflix", "Reading", "Music lover", "Gamer", "Traveller", "Foodie",
            "Photography", "Fashion", "Cooking", "Yoga", "Meditation", "Tech geek", "Bookworm",
            "Clean freak", "Pet lover", "Smoker", "Non-smoker", "Vegetarian",
            "Non-vegetarian", "Prefers quiet", "Likes guests over", "Study focused", "Chill roommate",
            "Funny", "Sarcastic", "Empathetic", "Friendly", "Supportive",
            "Open-minded", "Peaceful", "Low drama", "Respectful"
        ]
        selected_tags = st.multiselect("Select tags:", TAG_OPTIONS)

        submit = st.form_submit_button("Save Basic Info")

    # --- WHAT MATTERS SECTION (outside form for instant updates) ---
    st.markdown("### What *matters* to you?")

    if "select_all_matters" not in st.session_state:
        st.session_state["select_all_matters"] = False

    select_all = st.checkbox("Select All", value=st.session_state["select_all_matters"])
    st.session_state["select_all_matters"] = select_all

    # Dynamically show feature checkboxes
    matters = {}
    cols = st.columns(2)
    for i, f in enumerate(FEATURES):
        with cols[i % 2]:
            key = f"matters_{f}"
            if key not in st.session_state:
                st.session_state[key] = select_all
            # If Select All changed, update the state
            if select_all:
                st.session_state[key] = True
            matters[f] = st.checkbox(f, value=st.session_state[key], key=key)

    # --- SAVE EVERYTHING AFTER BOTH SECTIONS FILLED ---
    if submit:
        if not name.strip():
            st.error("Name required.")
        else:
            row = {
                "Name": name,
                "Age": age,
                "Gender": gender,
                "Cleanliness": cleanliness,
                "Noise": noise,
                "Smoking": 1 if smoking == "Yes" else 0,
                "SleepTime": sleeptime,
                "StudyTime": studytime,
                "PetFriendly": 1 if pet == "Yes" else 0,
                "Cooking": cooking,
                "Tags": ", ".join(selected_tags)
            }
            for f in FEATURES:
                row[f"Matters_{f}"] = 1 if st.session_state[f"matters_{f}"] else 0

            users = pd.concat([users, pd.DataFrame([row])], ignore_index=True)
            save_users(users)
            st.session_state["current_user"] = name
            st.success(f"Profile saved for {name} ✅")

# ---------------------------
# Get Recommendations Mode
# ---------------------------
if mode == "Get Recommendations":
    st.subheader("🔎 Find Matches")

    if users.empty:
        st.info("No users yet.")
    else:
        names = users["Name"].tolist()
        current = st.session_state.get("current_user", names[0])
        selected = st.selectbox("Select your profile", names, index=names.index(current))

        # Optional: Age filter
        min_age, max_age = st.slider("Filter by age range", 18, 60, (20, 35))

        if st.button("Show recommendations"):
            # Apply age filter before recommendation
            filtered_users = users[(users["Age"] >= min_age) & (users["Age"] <= max_age)]
            recs = recommend(filtered_users, selected)

            if recs.empty:
                st.info("No matches found.")
            else:
                st.markdown("### 💡 Your Best Matches")
                cols_per_row = 3
                user_row = users[users["Name"] == selected].iloc[0]

                # Ensure Tags column exists
                if "Tags" not in users.columns:
                    users["Tags"] = ""
                    recs["Tags"] = ""

                user_tags = set(str(user_row.get("Tags", "")).split(", ")) if user_row.get("Tags") else set()

                for i in range(0, len(recs), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for idx, col in enumerate(cols):
                        if i + idx < len(recs):
                            row = recs.iloc[i + idx]

                            # --- Compatibility Score ---
                            user_feats = np.array(user_row[FEATURES], dtype=float)
                            match_feats = np.array(row[FEATURES], dtype=float)
                            similarity = cosine_similarity([user_feats], [match_feats])[0][0]
                            compatibility_score = int((similarity + 1) * 50)  # scale -1..1 to 0..100

                            # --- Common Tags ---
                            match_tags = set(str(row.get("Tags", "")).split(", ")) if row.get("Tags") else set()
                            common_tags = user_tags.intersection(match_tags)
                            tags_html = ""
                            if common_tags:
                                tags_html = "<p style='color:#4CAF50;font-size:13px;'>💚 Common: " + ", ".join(common_tags) + "</p>"
                            else:
                                tags_html = "<p style='color:#999;font-size:13px;'>No common tags</p>"

                            with col:
                                st.markdown(
                                    f"""
                                    <div style="
                                        background-color:#2E2E2E;
                                        border-radius:18px;
                                        padding:20px;
                                        margin:10px;
                                        box-shadow:0 0 10px rgba(255,255,255,0.08);
                                        text-align:center;">
                                        <h3 style='color:#f0f0f0;margin-bottom:5px;'>{row['Name']}</h3>
                                        <p style='color:#ccc;margin:0;'>Age: {int(row['Age'])}</p>
                                        <p style='color:#aaa;margin:0;'>Gender: {row['Gender']}</p>
                                        <p style='color:#bbb;font-size:13px;'>🧹 Cleanliness: {row['Cleanliness']} | 🕓 Sleep: {row['SleepTime']}</p>
                                        <p style='color:#FFD700;font-weight:bold;'>💞 Compatibility: {compatibility_score}%</p>
                                        <p style='color:#888;font-size:12px;'>🏷️ {row.get('Tags', 'No tags')}</p>
                                        {tags_html}
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

