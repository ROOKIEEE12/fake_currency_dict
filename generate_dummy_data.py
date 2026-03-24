import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Configuration
DATASET_DIR = 'dataset'
IMG_SIZE = (224, 224)
SAMPLES_PER_CLASS = 50

def create_directory_structure():
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
    
    for category in ['Real', 'Fake']:
        path = os.path.join(DATASET_DIR, category)
        if not os.path.exists(path):
            os.makedirs(path)

def generate_image(category, idx):
    # Create a random background color image
    color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
    img = Image.new('RGB', IMG_SIZE, color=color)
    d = ImageDraw.Draw(img)
    
    # Add some random shapes/noise
    for _ in range(10):
        xy = [np.random.randint(0, i) for i in IMG_SIZE]
        xy.extend([np.random.randint(0, i) for i in IMG_SIZE])
        d.line(xy, fill=(0, 0, 0), width=3)

    # Add text to distinguish classes (simulating features)
    # real notes will have a green circle, fake will have a red square
    center = (IMG_SIZE[0]//2, IMG_SIZE[1]//2)
    
    if category == 'Real':
        # Draw green circle for "Real"
        d.ellipse([center[0]-40, center[1]-40, center[0]+40, center[1]+40], fill='green', outline='white', width=5)
        d.text((10, 10), "REAL CURRENCY", fill='white')
    else:
        # Draw red rectangle for "Fake"
        d.rectangle([center[0]-40, center[1]-40, center[0]+40, center[1]+40], fill='red', outline='white', width=5)
        d.text((10, 10), "FAKE CURRENCY", fill='white')

    # Save
    save_path = os.path.join(DATASET_DIR, category, f'img_{idx}.jpg')
    img.save(save_path)

def main():
    print("Generating synthetic dataset...")
    create_directory_structure()
    
    for i in range(SAMPLES_PER_CLASS):
        generate_image('Real', i)
        generate_image('Fake', i)
        
    print(f"Created {SAMPLES_PER_CLASS} images per class in '{DATASET_DIR}'.")
    print("You can now run 'python train_currency_model.py' to train the model.")

if __name__ == "__main__":
    main()
