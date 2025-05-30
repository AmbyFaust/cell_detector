import torch
import torch.nn as nn
from torchvision import transforms
import os
import torch.nn.functional as F

import cv2
import numpy as np

from skimage.measure import label, regionprops
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist


class MiniUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.down1 = self._block(3, 32, 3)
        self.down2 = self._block(32, 64, 3)
        self.down3 = self._block(64, 128, 3)

        # Decoder
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = self._block(128, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv2 = self._block(64, 32)

        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def _block(self, in_channels, out_channels, kernel=3):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        x1 = self.down1(x)
        x2 = F.max_pool2d(x1, 2)
        x2 = self.down2(x2)
        x3 = F.max_pool2d(x2, 2)
        x3 = self.down3(x3)

        # Decoder
        x = self.up1(x3)
        x = torch.cat([x, x2], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv2(x)

        return torch.sigmoid(self.out(x))


# Трансформы
transform = transforms.Compose([
    transforms.ToTensor(),
])

class Config:
    image_dir = "..\\..\\train\\original"
    mask_dir = "..\\..\\train\\train\\mask"
    batch_size = 6
    lr = 1e-4
    num_epochs = 50
    resize_to = (800, 600)
    save_path = "core\\cnn\\model_colorful_DICE-kernel-3-3-3-2-2-1.pth"


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MiniUNet()
model.load_state_dict(torch.load(Config.save_path, map_location=torch.device('cpu')))
model = model.to(device)
model.eval()


def get_pic_and_inference(model, transform, filepath):
    from PIL import Image
    import torch.nn.functional as F
    import cv2

    image = cv2.imread(filepath)

    im = Image.open(filepath).convert("RGB").resize(Config.resize_to)
    input_tensor = transform(im).unsqueeze(0).to(device)

    with torch.no_grad():
        model.eval()
        res = model(input_tensor).cpu()

        resized = F.interpolate(
            res,
            size=(1200, 1600),
            mode='bilinear',
            align_corners=False
        )
    return image, resized[0][0].numpy()


def detect_cell_centers(mask):
    labeled_mask = label(mask, connectivity=2)
    regions = regionprops(labeled_mask)
    centers =  np.array([(int(region.centroid[1]), int(region.centroid[0])) for region in regions])
    min_distance = 50
    if len(centers) == 0:
        return np.array([])

        # Создаем список для отфильтрованных центров
    filtered_centers = [centers[0]]

    for center in centers:
        # Проверяем расстояние до всех уже отфильтрованных центров
        distances = cdist([center], filtered_centers)
        if np.all(distances > min_distance):
            filtered_centers.append(center)

    return np.array(filtered_centers)


def cnn_detector(filepath) -> (int, str):
    image, res = get_pic_and_inference(model, transform, filepath)

    mask = cv2.GaussianBlur(
        src=res * 255,
        ksize=(3, 3),
        sigmaX=0,
    )
    _, white_mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)
    centers = detect_cell_centers(white_mask)
    for (x, y) in centers:
        cv2.circle(image, (x, y), 10, (0, 0, 255), -1)

    plt.imshow(image)
    result_dir_path = '..\\..\\result'
    cv_result_path = '..\\..\\result\\cnn_result.png'
    if not os.path.exists(result_dir_path):
        os.mkdir(result_dir_path)
    cv2.imwrite(cv_result_path, image)

    return cv_result_path, len(centers)
