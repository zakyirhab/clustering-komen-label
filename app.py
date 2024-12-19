import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Function to perform clustering
def perform_clustering(data, n_clusters):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(data['komentar'])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(tfidf_matrix)
    data['cluster'] = clusters
    return data, kmeans, vectorizer

# Streamlit UI
st.title("Komentar Clustering dan Labeling")
st.sidebar.header("Upload File")

# Upload file
uploaded_file = st.sidebar.file_uploader("Upload file Excel", type="xlsx")
if uploaded_file:
    data = pd.read_excel(uploaded_file)

    # Pastikan kolom sentimen ada
    if 'sentimen' not in data.columns:
        data['sentimen'] = None

    st.write("## Data yang diunggah:", data.head())

    # Input jumlah cluster
    n_clusters = st.sidebar.slider("Jumlah Cluster", min_value=2, max_value=50, value=5, step=1)

    # Perform clustering
    if st.sidebar.button("Lakukan Clustering"):
        unlabeled_data = data[data['sentimen'].isnull()]
        clustered_data, kmeans, vectorizer = perform_clustering(unlabeled_data, n_clusters)

        data.update(clustered_data)  # Update hasil clustering ke dataset utama
        st.write("## Data dengan Cluster:", clustered_data.head())

        # Display clusters
        for cluster_id in range(n_clusters):
            st.write(f"### Cluster {cluster_id}")
            cluster_data = data[(data['cluster'] == cluster_id) & (data['sentimen'].isnull())]
            st.write(cluster_data[['komentar', 'sentimen']])

            # Labeling
            bulk_label = st.text_input(f"Label untuk Cluster {cluster_id}", key=f"bulk_label_{cluster_id}")
            if st.button(f"Terapkan Label untuk Cluster {cluster_id}"):
                data.loc[(data['cluster'] == cluster_id) & (data['sentimen'].isnull()), 'sentimen'] = bulk_label

            # Individual labeling
            for idx, row in cluster_data.iterrows():
                individual_label = st.text_input(f"Label untuk komentar: {row['komentar']}", key=f"ind_label_{idx}")
                if st.button(f"Simpan Label untuk Komentar {idx}"):
                    data.at[idx, 'sentimen'] = individual_label

        # Save labeled data
        if st.button("Simpan Data"):
            with pd.ExcelWriter(uploaded_file.name, engine='openpyxl') as writer:
                data.to_excel(writer, index=False)
            st.success(f"Data berhasil disimpan ke {uploaded_file.name}!")

        st.write("## Data Terkini:", data.head())
