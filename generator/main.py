from config import generated_mask_path
from generator.generate import generate
from generator.preparator import prepare

def generate_images():
    prepare()
    return generate()