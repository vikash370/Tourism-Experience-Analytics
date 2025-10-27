import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_tourism_data.csv")

data = load_data()

st.title("🌍 Tourism Analytics Dashboard")

# Sidebar navigation
section = st.sidebar.radio("Choose Section", ["EDA", "Regression", "Classification", "Recommendation"])

# ------------------ EDA ------------------
if section == "EDA":
    st.header("📊 Exploratory Data Analysis")
    st.subheader("Rating Distribution")
    st.bar_chart(data["Rating"].value_counts().sort_index())

    if "VisitMode" in data.columns:
        st.subheader("Visit Mode Distribution")
        st.bar_chart(data["VisitMode"].value_counts())

    if "AttractionType" in data.columns:
        st.subheader("Average Rating by Attraction Type")
        avg_rating = data.groupby("AttractionType")["Rating"].mean().sort_values(ascending=False)
        st.bar_chart(avg_rating)

# ------------------ Regression ------------------
elif section == "Regression":
    st.header("📈 Predict Rating (Regression)")
    drop_cols = ["Rating", "TransactionId", "UserId", "AttractionId", "Attraction", "AttractionAddress"]
    X = data.drop(columns=[col for col in drop_cols if col in data.columns], errors="ignore")
    y = data["Rating"]

    # Encode if needed
    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write("**R² Score:**", r2_score(y_test, y_pred))
    st.write("**MSE:**", mean_squared_error(y_test, y_pred))

# ------------------ Classification ------------------
elif section == "Classification":
    st.header("🎯 Predict Visit Mode (Classification)")
    if "VisitMode" not in data.columns:
        st.warning("VisitMode column not found.")
    else:
        drop_cols = ["VisitMode", "TransactionId", "UserId", "AttractionId", "Attraction", "AttractionAddress", "Rating"]
        X = data.drop(columns=[col for col in drop_cols if col in data.columns], errors="ignore")
        y = data["VisitMode"]

        for col in X.select_dtypes(include="object").columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = GradientBoostingClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        st.write("**Accuracy:**", accuracy_score(y_test, y_pred))
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

# -------------------Recommendation-----------------

elif section == "Recommendation":
    st.header("🤝 Attraction Recommendation System")

    # 🔍 Show available columns for debugging
    st.write("Available columns in dataset:", data.columns.tolist())

    # ✅ Collaborative Filtering
    st.subheader("Collaborative Filtering")
    required_cols = ["UserId", "Attraction", "Rating"]
    missing = [col for col in required_cols if col not in data.columns]

    if missing:
        st.error(f"Missing columns for collaborative filtering: {missing}")
    else:
        try:
            user_item_matrix = data.pivot_table(index="UserId", columns="Attraction", values="Rating", fill_value=0)

            if user_item_matrix.shape[0] < 2 or user_item_matrix.shape[1] < 2:
                st.warning("Not enough data to compute user similarity.")
            else:
                user_similarity = cosine_similarity(user_item_matrix)
                user_sim_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

                def recommend_collab(user_id, top_n=5):
                    if user_id not in user_sim_df.index:
                        return []
                    similar_users = user_sim_df[user_id].sort_values(ascending=False)[1:]
                    top_user = similar_users.index[0]
                    user_rated = user_item_matrix.loc[user_id]
                    top_user_rated = user_item_matrix.loc[top_user]
                    recs = top_user_rated[(user_rated == 0) & (top_user_rated > 0)].sort_values(ascending=False)
                    return recs.head(top_n).index.tolist()

                user_input = st.number_input("Enter User ID", min_value=0, step=1)
                if st.button("Get Collaborative Recommendations"):
                    recs = recommend_collab(user_input)
                    if recs:
                        st.write("Recommended Attractions:", recs)
                    else:
                        st.warning("No recommendations found for this user.")
        except Exception as e:
            st.error(f"Collaborative filtering failed: {e}")

    # ✅ Content-Based Filtering
    st.subheader("Content-Based Filtering")
    content_cols = ["AttractionType", "Country", "Continent"]
    optional_cols = ["Region"]
    content_cols += [col for col in optional_cols if col in data.columns]

    missing_content = [col for col in content_cols if col not in data.columns]
    if missing_content:
        st.warning(f"Some content columns are missing: {missing_content}")
    else:
        try:
            for col in content_cols:
                data[col] = LabelEncoder().fit_transform(data[col].astype(str))

            attraction_profiles = data.groupby("Attraction")[content_cols].mean()

            if attraction_profiles.shape[0] < 2:
                st.warning("Not enough attractions to compute similarity.")
            else:
                attraction_sim = cosine_similarity(attraction_profiles)
                attraction_sim_df = pd.DataFrame(attraction_sim, index=attraction_profiles.index, columns=attraction_profiles.index)

                attraction_input = st.selectbox("Choose an Attraction", attraction_sim_df.index.tolist())
                if st.button("Get Similar Attractions"):
                    similar = attraction_sim_df[attraction_input].sort_values(ascending=False)[1:6]
                    st.write("Similar Attractions:", similar.index.tolist())
        except Exception as e:
            st.error(f"Content-based filtering failed: {e}")
