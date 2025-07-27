import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans

# Load data and models
@st.cache_data
def load_data():
    df = pd.read_csv("cld_data_final.csv")
    with open("encoder1.pkl", "rb") as f:
        encoder = pickle.load(f)
    with open("kmeans_model_new.pkl", "rb") as f:
        kmeans = pickle.load(f)
    return df, encoder, kmeans

df, encoder, kmeans = load_data()

st.title("🍽️ Swiggy Restaurant Recommendation")


st.sidebar.header("Filter Your Preferences")
main_city = st.sidebar.selectbox("Select City", df['main_city'].dropna().unique())
cuisine = st.sidebar.selectbox("Select Cuisine", df['cuisine'].dropna().unique())
rating = st.sidebar.slider("Minimum Rating", 1.0, 5.0, 4.0)
cost = st.sidebar.slider("Maximum Cost (₹)", 100, 2000, 500)


filtered_df = df[(df['main_city'] == main_city) & (df['cuisine'] == cuisine)]

if filtered_df.empty:
    st.warning("⚠️ No restaurants found with this combination. Try changing filters.")
else:
    # Encode input for clustering
    input_data = encoder.transform([[main_city, cuisine]])
    input_df = pd.DataFrame(input_data, columns=encoder.get_feature_names_out())
    input_df['rating'] = rating
    input_df['rating_count'] = 100  # default
    input_df['cost'] = cost

    # Predict cluster
    cluster_id = kmeans.predict(input_df)[0]

    st.subheader(f"🔍 Showing Restaurants from Cluster #{cluster_id}")
    results = df[df['cluster'] == cluster_id][['name', 'city', 'main_city', 'cuisine', 'rating', 'cost', 'address']].sample(n=5)

    st.dataframe(results)

