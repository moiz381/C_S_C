import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# App Title
st.title("🧠 Customer Segmentation using Machine Learning")
st.markdown("Built by Hafiz Moiz Ali | A professional ML Project")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("synthetic_mall_customers.csv")

df = load_data()

# Sidebar
st.sidebar.header("Select Options")
show_data = st.sidebar.checkbox("Show Raw Dataset", value=True)
show_eda = st.sidebar.checkbox("Show Data Visualizations")
run_clustering = st.sidebar.button("Run KMeans Clustering")

# Show Dataset
if show_data:
    st.subheader("📊 Dataset Preview")
    st.write(df.head())

# Encode Gender
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

# Feature Selection & Scaling
features = ['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Show EDA
if show_eda:
    st.subheader("📈 Exploratory Data Analysis")
    for col in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)

# Clustering
if run_clustering:
    st.subheader("🧪 KMeans Clustering & Visualization")

    # KMeans
    kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # PCA for Visualization
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    df['PCA1'] = components[:, 0]
    df['PCA2'] = components[:, 1]

    # Scatter Plot
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='Set1', ax=ax)
    ax.set_title("Customer Segments (via PCA)")
    st.pyplot(fig)

    # Cluster Insights
    st.subheader("🔍 Cluster Insights")
    cluster_summary = df.groupby('Cluster')[features].mean().round(1)
    st.write(cluster_summary)
