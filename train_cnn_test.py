import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATASET_DIR = r'archive\data\data'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

if not os.path.exists(DATASET_DIR):
    # Try alternate if it's generate_dummy_data
    if os.path.exists('dataset'):
        DATASET_DIR = 'dataset'
    else:
        print("Dataset not found!")
        exit(1)

print("Loading dataset from", DATASET_DIR)
datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)

train_generator = datagen.flow_from_directory(
    DATASET_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', subset='training'
)

val_generator = datagen.flow_from_directory(
    DATASET_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', subset='validation'
)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Training CNN model...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5
)

# Evaluate
loss, acc = model.evaluate(val_generator)
print(f"Validation Accuracy: {acc*100:.2f}%")

model.save("currency_model_cnn.h5")
print("Model saved to currency_model_cnn.h5")
