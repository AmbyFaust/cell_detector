import cv2
import numpy as np
import os
import tqdm
from matplotlib import pyplot as plt
from config import base_dir, output_cells_dir, output_background_dir, output_filled_cells_dir, \
    output_filled_background_dir, original_dir, masked_dir
from generator.split_filled import get_filled_cells, get_filled_back
from generator.split_transparent import split_transparent

def prepare():
    """
    Разделяет картинки на фон и клетки, сохраняет все это дело
    """
    os.makedirs(output_cells_dir, exist_ok=True)
    os.makedirs(output_background_dir, exist_ok=True)

    os.makedirs(output_filled_cells_dir, exist_ok=True)
    os.makedirs(output_filled_background_dir, exist_ok=True)

    split_transparent(original_dir=original_dir,
                      masked_dir=masked_dir,
                      output_background_dir=output_background_dir,
                      output_cells_dir=output_cells_dir)

    get_filled_cells(original_dir=original_dir,
                     masked_dir=masked_dir,
                     output_filled_cells_dir=output_filled_cells_dir)

    get_filled_back(original_dir=original_dir,
                    masked_dir=masked_dir,
                    output_filled_background_dir=output_filled_background_dir)