import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Configuration
DATASET_DIR = r'archive\data\data'  # Pointing to the new dataset
MODEL_PATH = 'currency_model.pkl'
IMG_SIZE = (64, 64)  # Smaller size for standard ML

print("Loading dataset...")

data = []
labels = []
# 'fake' folder maps to label 0, 'real' folder maps to label 1
class_mapping = {'fake': 0, 'real': 1}

for category, label in class_mapping.items():
    category_path = os.path.join(DATASET_DIR, category)
    if not os.path.exists(category_path):
        print(f"Error: {category_path} does not exist!")
        exit(1)
    
    # Iterate through subfolders (10, 20, 50, etc.)
    for denomination in os.listdir(category_path):
        denom_path = os.path.join(category_path, denomination)
        
        if os.path.isdir(denom_path):
            print(f"Loading {category} - {denomination}...")
            for img_name in os.listdir(denom_path):
                try:
                    img_path = os.path.join(denom_path, img_name)
                    # Check if it's an image file
                    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue
                        
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize(IMG_SIZE)
                    img_array = np.array(img).flatten()  # Flatten for ML model
                    
                    data.append(img_array)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading {img_name}: {e}")

X = np.array(data)
y = np.array(labels)

print(f"Loaded {len(X)} images total.")

if len(X) == 0:
    print("No images found! Check the dataset path.")
    exit(1)

# Split
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
print("Training Random Forest Classifier...")
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) # parallel jobs
clf.fit(X_train, y_train)

# Evaluate
print("Evaluating model...")
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

# Save
joblib.dump(clf, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
