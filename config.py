import os

base_dir = "C:/Users/User/lec_2/blood_cells_dataset/BCCD/"

original_dir = os.path.join(base_dir, "train/original")
masked_dir = os.path.join(base_dir, "train/mask")
output_cells_dir = os.path.join(base_dir, "transparent/cell_patches")
output_background_dir = os.path.join(base_dir, "transparent/background_patches")

output_filled_cells_dir = os.path.join(base_dir, "filled/cell_patches")
output_filled_background_dir = os.path.join(base_dir, "filled/background_patches")

generated_orig_path = os.path.join(base_dir, "generated/original")
generated_mask_path = os.path.join(base_dir, "generated/mask")

min_circles=20
max_circles=100
min_radius=30
max_radius=70
blue_range=(100, 200)
purple_range=(50, 150)
color_variation = 20