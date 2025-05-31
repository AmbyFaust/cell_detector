import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

import os

base_path = "C:/Users/User/lec_2/blood_cells_dataset/BCCD/train/original"
image_path = "5f783f43-0090-4f8a-bd75-9562f4aa2dd5.png"
image_path = os.path.join(base_path, image_path)


def ml_detector(image_path: str):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (400, 300), interpolation=cv2.INTER_AREA)

    # 2. Улучшение контраста (CLAHE в LAB)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l)
    enhanced_lab = cv2.merge((enhanced_l, a, b))
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

    # 3. LBP текстура
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    lbp_8bit = np.uint8((lbp / lbp.max()) * 255)

    denoised = cv2.bilateralFilter(lbp_8bit, 9, 90, 90)
    smoothed = cv2.GaussianBlur(denoised, (5, 5), 0)

    mask = cv2.inRange(smoothed, 100, 125)  # Диапазон значений колец

    y, x = np.where(mask > 0)
    features = np.column_stack((x, y))

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    min_clusters = 10
    max_clusters = 100
    while True:
        print(min_clusters, max_clusters)
        if max_clusters - min_clusters < 4:
            break
        cl1 = (max_clusters - min_clusters) // 3 + min_clusters
        cl2 = (max_clusters - min_clusters) * 2 // 3 + min_clusters
        print("\t", cl1, cl2)

        kmeans1 = KMeans(n_clusters=cl1, random_state=42)
        kmeans1.fit(features_scaled)
        score1 = silhouette_score(features_scaled, kmeans1.labels_)

        kmeans2 = KMeans(n_clusters=cl2, random_state=42)
        kmeans2.fit(features_scaled)
        score2 = silhouette_score(features_scaled, kmeans2.labels_)

        if score1 > score2:
            max_clusters = cl2
        else:
            min_clusters = cl1

    n_clusters = max_clusters

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)

    masked_img = cv2.bitwise_and(image, image, mask=mask)

    result = image.copy()

    for cluster_id in range(n_clusters):
        cluster_points = features[labels == cluster_id]

        if len(cluster_points) < 10:
            continue

        # Вычисляем центр и радиус кластера
        center = np.mean(cluster_points, axis=0).astype(int)
        radius = int(np.sqrt(len(cluster_points) / np.pi))  # Примерный радиус

        # Рисуем окружность и номер
        cv2.circle(result, tuple(center), radius, (0, 255, 0), 2)
        cv2.putText(result, str(cluster_id + 1), tuple(center),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    result_dir_path = '..\\..\\result'
    ml_result_path = '..\\..\\result\\ml_result.png'
    if not os.path.exists(result_dir_path):
        os.mkdir(result_dir_path)
    cv2.imwrite(ml_result_path, result)

    return ml_result_path, n_clusters
