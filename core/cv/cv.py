import cv2
import numpy as np

from skimage.measure import label, regionprops
from sklearn.cluster import DBSCAN


def detect_cell_centers(mask):
    labeled_mask = label(mask, connectivity=2)
    regions = regionprops(labeled_mask)
    return np.array([(int(region.centroid[1]), int(region.centroid[0])) for region in regions])


def cv_detector(image_path, mask_path, result_path) -> (int, str):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(image_path)
    mask = cv2.GaussianBlur(
        src=mask,
        ksize=(3, 3),
        sigmaX=0,
    )
    _, white_mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)
    centers = detect_cell_centers(white_mask)
    for (x, y) in centers:
        cv2.circle(image, (x, y), 10, (0, 0, 255), -1)

    cv2.imwrite(result_path, image)
    return len(centers), result_path




mask_path = 'C:\\Projects\\cell_detector\\data\\test\\mask\\e3c1442a-717f-41dd-bf97-81e1233ac9fa.png'
file_path = 'C:\\Projects\\cell_detector\\data\\test\\original\\e3c1442a-717f-41dd-bf97-81e1233ac9fa.png'
result_path = '..\\..\\result\\cv_result.png'

a, b = cv_detector(file_path, mask_path, result_path)
print(a)