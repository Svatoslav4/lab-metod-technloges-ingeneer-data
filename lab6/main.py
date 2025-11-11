# ============================================
# Лабораторна: Кластеризація та розпаралелення
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from time import time
from multiprocessing import Pool
from joblib import Parallel, delayed

# -----------------------------
# Глобальні функції
# -----------------------------

def safe_silhouette(X, labels):
    """Обчислення Silhouette Score без помилок"""
    if len(set(labels)) > 1 and -1 not in set(labels):
        return silhouette_score(X, labels)
    else:
        return np.nan

def kmeans_fit(k, X_scaled):
    """Функція для multiprocessing/joblib"""
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X_scaled)
    return model.inertia_

# -----------------------------
# Основна функція
# -----------------------------
def main():
    # -----------------------------
    # 1. Підготовка даних
    # -----------------------------
    print("=== 1. Підготовка даних ===")
    df = pd.read_csv("Mall_Customers.csv")

    print("\nПерші рядки даних:")
    print(df.head())
    print("\nІнформація про дані:")
    print(df.info())
    print("\nКількість пропущених значень:")
    print(df.isnull().sum())

    # Видалимо ID і кодуємо стать
    if 'CustomerID' in df.columns:
        df = df.drop('CustomerID', axis=1)
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

    # Масштабування
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # -----------------------------
    # 2. Кластеризація
    # -----------------------------
    print("\n=== 2. Кластеризація ===")
    inertia = []
    K_range = range(2, 11)
    for k in K_range:
        model = KMeans(n_clusters=k, random_state=42)
        model.fit(X_scaled)
        inertia.append(model.inertia_)

    plt.figure(figsize=(6, 4))
    plt.plot(K_range, inertia, 'o-')
    plt.title("Метод ліктя (Elbow Method)")
    plt.xlabel("Кількість кластерів")
    plt.ylabel("Inertia")
    plt.show()

    n_clusters = 5

    # --- KMeans ---
    print("\nЗапуск KMeans...")
    start = time()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels_kmeans = kmeans.fit_predict(X_scaled)
    time_kmeans = time() - start
    print(f"KMeans завершено за {time_kmeans:.3f} c")

    # --- DBSCAN ---
    print("\nЗапуск DBSCAN...")
    start = time()
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels_dbscan = dbscan.fit_predict(X_scaled)
    time_dbscan = time() - start
    print(f"DBSCAN завершено за {time_dbscan:.3f} c")

    # --- Agglomerative ---
    print("\nЗапуск Agglomerative Clustering...")
    start = time()
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    labels_agg = agg.fit_predict(X_scaled)
    time_agg = time() - start
    print(f"Agglomerative Clustering завершено за {time_agg:.3f} c")

    # -----------------------------
    # 3. Оцінка якості кластерів
    # -----------------------------
    print("\n=== 3. Оцінка якості кластеризації ===")
    sil_kmeans = safe_silhouette(X_scaled, labels_kmeans)
    sil_dbscan = safe_silhouette(X_scaled, labels_dbscan)
    sil_agg = safe_silhouette(X_scaled, labels_agg)

    print(f"Silhouette KMeans: {sil_kmeans:.3f}")
    print(f"Silhouette DBSCAN: {sil_dbscan}")
    print(f"Silhouette Agglomerative: {sil_agg:.3f}")

    df['KMeans_Cluster'] = labels_kmeans
    df['DBSCAN_Cluster'] = labels_dbscan
    df['Agg_Cluster'] = labels_agg

    # -----------------------------
    # 4. Візуалізація
    # -----------------------------
    print("\n=== 4. Візуалізація кластерів ===")
    pca = PCA(2)
    X_pca = pca.fit_transform(X_scaled)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels_kmeans)
    axs[0].set_title("KMeans")
    axs[1].scatter(X_pca[:, 0], X_pca[:, 1], c=labels_dbscan)
    axs[1].set_title("DBSCAN")
    axs[2].scatter(X_pca[:, 0], X_pca[:, 1], c=labels_agg)
    axs[2].set_title("Agglomerative")
    plt.show()

    linked = linkage(X_scaled, 'ward')
    plt.figure(figsize=(10, 6))
    dendrogram(linked)
    plt.title("Dendrogram (Agglomerative Clustering)")
    plt.show()

    # -----------------------------
    # 5. Розпаралелення
    # -----------------------------
    print("\n=== 5. Розпаралелення процесу кластеризації ===")

    start = time()
    with Pool(processes=4) as pool:
        inertia_parallel = pool.starmap(kmeans_fit, [(k, X_scaled) for k in range(2, 11)])
    time_mp = time() - start
    print(f"Multiprocessing час: {time_mp:.3f} c")

    start = time()
    results = Parallel(n_jobs=4)(delayed(kmeans_fit)(k, X_scaled) for k in range(2, 11))
    time_joblib = time() - start
    print(f"Joblib час: {time_joblib:.3f} c")

    # -----------------------------
    # 6. Порівняння результатів
    # -----------------------------
    print("\n=== 6. Порівняння часу виконання ===")
    comparison = pd.DataFrame({
        'Алгоритм': ['KMeans', 'DBSCAN', 'Agglomerative', 'KMeans + multiprocessing', 'KMeans + joblib'],
        'Час виконання (с)': [time_kmeans, time_dbscan, time_agg, time_mp, time_joblib],
        'Silhouette Score': [sil_kmeans, sil_dbscan, sil_agg, '-', '-']
    })
    print(comparison)

    plt.figure(figsize=(8, 4))
    sns.barplot(data=comparison, x='Алгоритм', y='Час виконання (с)')
    plt.title("Порівняння часу кластеризації")
    plt.xticks(rotation=20)
    plt.show()

    print("\n=== Роботу завершено успішно ✅ ===")

# -----------------------------
# Запуск
# -----------------------------
if __name__ == "__main__":
    main()
