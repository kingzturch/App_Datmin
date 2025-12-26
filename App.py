%%writefile app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Asset Clustering Dashboard",
    layout="wide"
)

st.title("📊 Asset Clustering & Insight Dashboard")

# =====================================================
# LOAD DATA & PREPROCESSING (CACHED)
# =====================================================
@st.cache_data
def load_and_preprocess_data():
    try:
        df = pd.read_csv("/content/drive/MyDrive/Datmin/Ba'da UTS/FPDatmin_cleaned2.csv")
    except FileNotFoundError:
        st.error("❌ Dataset tidak ditemukan. Pastikan path CSV benar.")
        st.stop()

    clustering_features = [
        'AcquisitionValue',
        'LifePlan',
        'MonthlyDepreValue',
        'NBV',
        'DepreAccumulation',
        'TotalDepreciation',
        'DepreRecords'
    ]

    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df[clustering_features]),
        columns=clustering_features
    )

    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(df_scaled)

    df_pca = df_scaled.copy()
    df_pca['pca1'] = pca_components[:, 0]
    df_pca['pca2'] = pca_components[:, 1]

    # =====================
    # OPTIMAL CLUSTERING
    # =====================
    optimal_k_kmeans = 5
    df_pca['kmeans_cluster'] = KMeans(
        n_clusters=optimal_k_kmeans,
        random_state=42,
        n_init=10
    ).fit_predict(df_scaled)

    eps_dbscan = 0.5
    min_samples_dbscan = 10
    df_pca['dbscan_cluster'] = DBSCAN(
        eps=eps_dbscan,
        min_samples=min_samples_dbscan
    ).fit_predict(df_scaled)

    n_clusters_hac = 2
    linkage_method_hac = "average"
    df_pca['hac_cluster'] = AgglomerativeClustering(
        n_clusters=n_clusters_hac,
        linkage=linkage_method_hac
    ).fit_predict(df_scaled)

    return (
        df, df_scaled, df_pca,
        clustering_features,
        optimal_k_kmeans,
        eps_dbscan,
        min_samples_dbscan,
        n_clusters_hac,
        linkage_method_hac
    )


df, df_scaled, df_pca, clustering_features, optimal_k_kmeans, eps_dbscan, min_samples_dbscan, n_clusters_hac, linkage_method_hac = load_and_preprocess_data()

# =====================================================
# CLUSTER DESCRIPTION FUNCTION
# =====================================================
def get_cluster_description(algo, cluster_id):

    descriptions = {

        "kmeans": {
            0: "Aset bernilai rendah dengan masa pakai standar dan tingkat depresiasi minimal. Umumnya merupakan aset operasional rutin atau aset yang relatif baru dengan catatan depresiasi yang masih sedikit.",
            1: "Aset strategis bernilai sangat tinggi dengan masa pakai sangat panjang serta riwayat depresiasi yang lengkap. Klaster ini merepresentasikan investasi jangka panjang bernilai besar.",
            2: "Aset bernilai tinggi dengan masa pakai panjang dan depresiasi moderat. Karakteristiknya menunjukkan aset penting yang masih aktif digunakan.",
            3: "Aset bernilai sangat tinggi tanpa depresiasi. Biasanya merupakan aset baru, aset dalam proses pencatatan, atau aset non-depresiasi seperti tanah.",
            4: "Aset bernilai tinggi dengan depresiasi signifikan. Menunjukkan aset lama yang telah digunakan cukup lama namun masih memiliki nilai buku yang besar."
        },

        "dbscan": {
            0: "Aset bernilai menengah dengan masa pakai panjang dan depresiasi stabil. Aset dalam klaster ini memiliki pola penggunaan yang konsisten.",
            1: "Aset bernilai rendah dengan umur relatif pendek dan depresiasi rendah. Umumnya merupakan aset operasional kecil atau baru.",
            -1: "Aset noise atau anomali. Aset ini memiliki karakteristik yang sangat berbeda dari kelompok lainnya dan perlu dianalisis secara khusus."
        },

        "hac": {
            0: "Aset bernilai menengah dengan masa pakai standar dan tingkat depresiasi rendah. Menunjukkan aset yang masih berada pada fase awal atau menengah penggunaan.",
            1: "Aset strategis bernilai sangat tinggi dengan masa pakai panjang dan depresiasi signifikan. Klaster ini mencerminkan aset premium organisasi."
        }
    }

    return descriptions.get(algo, {}).get(cluster_id, "Deskripsi klaster tidak tersedia.")

# =====================================================
# SIDEBAR MENU
# =====================================================
menu = st.sidebar.radio("📌 PILIH MENU", [
    "K-Means Clustering",
    "DBSCAN Clustering",
    "Hierarchical Clustering",
    "Insight Klaster (Ringkasan)",
    "Visualisasi Tambahan",
    "Filter Data per Klaster",
    "Download Hasil"
])

# =====================================================
# K-MEANS
# =====================================================
if menu == "K-Means Clustering":
    st.header("🔹 K-Means Clustering")

    k = st.sidebar.slider("Jumlah Klaster (K)", 2, 10, optimal_k_kmeans)
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(df_scaled)
    df_pca['kmeans_display'] = labels

    fig, ax = plt.subplots(figsize=(9,6))
    sns.scatterplot(data=df_pca, x='pca1', y='pca2', hue='kmeans_display', palette='viridis', ax=ax)
    ax.set_title(f"K-Means Clustering (K={k}) - PCA")
    st.pyplot(fig)

      # =========================
    # VALIDASI KLASTER K-MEANS
    # =========================
    sil_score = silhouette_score(df_scaled, labels)
    dbi_score = davies_bouldin_score(df_scaled, labels)

    st.markdown("### 📊 Hasil Validasi Klaster")
    st.write(f"**Silhouette Score** : `{sil_score:.4f}`")
    st.write(f"**Davies–Bouldin Index** : `{dbi_score:.4f}`")

    st.info(
        "Silhouette Score yang lebih mendekati 1 menunjukkan pemisahan klaster yang baik, "
        "sedangkan nilai Davies–Bouldin yang lebih kecil menunjukkan klaster yang lebih kompak dan terpisah."
    )


# =====================================================
# DBSCAN
# =====================================================
elif menu == "DBSCAN Clustering":
    st.header("🔹 DBSCAN Clustering")

    eps = st.sidebar.slider("eps", 0.1, 2.0, eps_dbscan)
    min_samp = st.sidebar.slider("min_samples", 3, 50, min_samples_dbscan)

    labels = DBSCAN(eps=eps, min_samples=min_samp).fit_predict(df_scaled)
    df_pca['dbscan_display'] = labels

    fig, ax = plt.subplots(figsize=(9,6))
    sns.scatterplot(data=df_pca, x='pca1', y='pca2', hue='dbscan_display', palette='plasma', ax=ax)
    ax.set_title("DBSCAN Clustering - PCA")
    st.pyplot(fig)

    # =========================
    # VALIDASI KLASTER DBSCAN
    # =========================
    unique_labels = set(labels)

    # Validasi hanya jika klaster lebih dari 1 dan tidak semuanya noise
    if len(unique_labels) > 1 and not (len(unique_labels) == 1 and -1 in unique_labels):

        valid_mask = labels != -1
        sil_score = silhouette_score(df_scaled[valid_mask], labels[valid_mask])
        dbi_score = davies_bouldin_score(df_scaled[valid_mask], labels[valid_mask])

        st.markdown("### 📊 Hasil Validasi Klaster")
        st.write(f"**Silhouette Score** : `{sil_score:.4f}`")
        st.write(f"**Davies–Bouldin Index** : `{dbi_score:.4f}`")

        st.info(
            "Evaluasi dilakukan tanpa menyertakan data noise (-1). "
            "Nilai ini menunjukkan kualitas klaster inti yang terbentuk oleh DBSCAN."
        )
    else:
        st.warning(
            "Validasi klaster tidak dapat dilakukan karena DBSCAN hanya membentuk satu klaster "
            "atau seluruh data teridentifikasi sebagai noise."
        )

# =====================================================
# HIERARCHICAL
# =====================================================
elif menu == "Hierarchical Clustering":
    st.header("🔹 Hierarchical Clustering")

    fig, ax = plt.subplots(figsize=(14,7))
    dendrogram(linkage(df_scaled, method=linkage_method_hac), ax=ax)
    ax.set_title("Dendrogram - Hierarchical Clustering")
    st.pyplot(fig)
    # =========================
    # VALIDASI KLASTER HIERARCHICAL
    # =========================
    labels = df_pca['hac_cluster']

    sil_score = silhouette_score(df_scaled, labels)
    dbi_score = davies_bouldin_score(df_scaled, labels)

    st.markdown("### 📊 Hasil Validasi Klaster")
    st.write(f"**Silhouette Score** : `{sil_score:.4f}`")
    st.write(f"**Davies–Bouldin Index** : `{dbi_score:.4f}`")

    st.info(
        "Nilai evaluasi ini menunjukkan kualitas klaster hasil Hierarchical Clustering "
        "berdasarkan struktur jarak antar data."
    )

# =====================================================
# INSIGHT KLASTER
# =====================================================
elif menu == "Insight Klaster (Ringkasan)":
    st.header("📌 Insight & Interpretasi Klaster")

    algo = st.sidebar.selectbox("Pilih Algoritma", ["K-Means", "DBSCAN", "Hierarchical"])

    if algo == "K-Means":
        col, key = "kmeans_cluster", "kmeans"
    elif algo == "DBSCAN":
        col, key = "dbscan_cluster", "dbscan"
    else:
        col, key = "hac_cluster", "hac"

    for c in sorted(df_pca[col].unique()):
        st.subheader(f"🔹 Klaster {c}")
        st.write(get_cluster_description(key, c))

        temp = df.copy()
        temp['cluster'] = df_pca[col]
        stats = temp[temp['cluster'] == c][clustering_features].mean().round(2)

        st.dataframe(stats.to_frame("Rata-rata"))

# =====================================================
# VISUALISASI TAMBAHAN
# =====================================================
elif menu == "Visualisasi Tambahan":
    st.header("📌 Visualisasi Tambahan")

    fig, ax = plt.subplots(figsize=(10,7))
    sns.heatmap(df[clustering_features].corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# =====================================================
# FILTER DATA
# =====================================================
elif menu == "Filter Data per Klaster":
    algo = st.sidebar.selectbox("Algoritma", ["K-Means", "DBSCAN", "Hierarchical"])

    col = {
        "K-Means": "kmeans_cluster",
        "DBSCAN": "dbscan_cluster",
        "Hierarchical": "hac_cluster"
    }[algo]

    cluster_id = st.sidebar.selectbox("Pilih Klaster", sorted(df_pca[col].unique()))
    st.dataframe(df.loc[df_pca[col] == cluster_id])

# =====================================================
# DOWNLOAD
# =====================================================
elif menu == "Download Hasil":
    algo = st.sidebar.selectbox("Algoritma", ["K-Means", "DBSCAN", "Hierarchical"])

    col = {
        "K-Means": "kmeans_cluster",
        "DBSCAN": "dbscan_cluster",
        "Hierarchical": "hac_cluster"
    }[algo]

    output = df.copy()
    output['cluster'] = df_pca[col]

    st.download_button(
        "Download CSV",
        output.to_csv(index=False).encode("utf-8"),
        file_name=f"hasil_clustering_{algo}.csv"
    )

st.sidebar.markdown("---")
st.sidebar.markdown("👥 Autor: Kelompok 12")
