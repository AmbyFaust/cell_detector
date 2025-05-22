import cv2
import numpy as np

from skimage.measure import label, regionprops
from sklearn.cluster import DBSCAN


def detect_cell_centers(mask):
    labeled_mask = label(mask, connectivity=2)
    regions = regionprops(labeled_mask)
    return np.array([(int(region.centroid[1]), int(region.centroid[0])) for region in regions])


def cv_detector(image_path, mask_path) -> (int, str):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(image_path)

    centers = detect_cell_centers(mask)

    clustering = DBSCAN(eps=10, min_samples=1).fit(centers)
    count_cells = len(np.unique(clustering.labels_))

    result_path = '..\\..\\result\\cv_result.png'

    for (x, y) in centers:
        cv2.circle(image, (x, y), 10, (0, 0, 255), -1)

    cv2.imwrite(result_path, image)
    return count_cells, result_path

