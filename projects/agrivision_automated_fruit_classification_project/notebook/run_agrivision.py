#!/usr/bin/env python3
"""
AgriVision: Automated Multi-Class Fruit Classification and Quality Control System
Python script version - can be run directly without Jupyter
"""

# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Deep Learning libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, ResNet50, VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Evaluation metrics
from sklearn.metrics import classification_report, confusion_matrix
import itertools

print("=" * 70)
print("AgriVision: Automated Multi-Class Fruit Classification System")
print("=" * 70)
print("\nTensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))
print()

# ============================================================================
# Phase 1: Data Exploration & Preprocessing
# ============================================================================
print("\n" + "=" * 70)
print("PHASE 1: DATA EXPLORATION & PREPROCESSING")
print("=" * 70)

# Set data paths
BASE_DIR = Path('../data')
TRAIN_DIR = BASE_DIR / 'train'
VALID_DIR = BASE_DIR / 'valid'
TEST_DIR = BASE_DIR / 'test'

# Identify Classes
class_names = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
print("\nCLASSES IDENTIFIED:")
for i, class_name in enumerate(class_names, 1):
    print(f"{i}. {class_name}")
print(f"\nTotal number of classes: {len(class_names)}")

# Visualize Samples
print("\nGenerating sample images visualization...")
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle('Sample Images from Each Fruit Class', fontsize=16, fontweight='bold')

for idx, class_name in enumerate(class_names):
    row = idx // 5
    col = idx % 5
    class_dir = TRAIN_DIR / class_name
    image_files = list(class_dir.glob('*.jpg'))
    
    if image_files:
        img_path = image_files[0]
        img = mpimg.imread(img_path)
        axes[row, col].imshow(img)
        axes[row, col].set_title(class_name.title(), fontsize=12, fontweight='bold')
        axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('sample_images.png', dpi=150, bbox_inches='tight')
print("Saved: sample_images.png")
plt.close()

# Distribution Plot
print("\nGenerating distribution plot...")
train_counts = {}
for class_name in class_names:
    class_dir = TRAIN_DIR / class_name
    train_counts[class_name] = len(list(class_dir.glob('*.jpg')))

plt.figure(figsize=(14, 6))
bars = plt.bar(train_counts.keys(), train_counts.values(), color='steelblue', edgecolor='black')
plt.title('Distribution of Images per Class in Training Set', fontsize=14, fontweight='bold')
plt.xlabel('Fruit Class', fontsize=12)
plt.ylabel('Number of Images', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('distribution_plot.png', dpi=150, bbox_inches='tight')
print("Saved: distribution_plot.png")
plt.close()

# Print summary statistics
total_images = sum(train_counts.values())
print("\nTRAINING SET STATISTICS:")
print(f"Total images: {total_images}")
print(f"Average per class: {total_images / len(class_names):.1f}")
print(f"Min images: {min(train_counts.values())}")
print(f"Max images: {max(train_counts.values())}")

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = len(class_names)

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

valid_test_datagen = ImageDataGenerator(rescale=1./255)

print("\nData augmentation configured successfully!")
print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Number of classes: {NUM_CLASSES}")

# Create data generators
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=42
)

valid_generator = valid_test_datagen.flow_from_directory(
    VALID_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    seed=42
)

test_generator = valid_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    seed=42
)

print("\nDATA GENERATORS CREATED:")
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {valid_generator.samples}")
print(f"Test samples: {test_generator.samples}")

# ============================================================================
# Phase 2: Model Development - Transfer Learning
# ============================================================================
print("\n" + "=" * 70)
print("PHASE 2: MODEL DEVELOPMENT - TRANSFER LEARNING")
print("=" * 70)

# Load pre-trained MobileNetV2
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False
print("\nBase model (MobileNetV2) loaded and frozen!")

# Build model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

print("\nMODEL ARCHITECTURE:")
model.summary()

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel compiled successfully!")

# Define callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'best_fruit_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=0.00001,
    verbose=1
)

callbacks = [early_stopping, checkpoint, reduce_lr]

# Train the model
print("\n" + "=" * 70)
print("STARTING TRAINING...")
print("=" * 70)
EPOCHS = 20

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=valid_generator,
    callbacks=callbacks,
    verbose=1
)

print("\nTraining completed!")

# ============================================================================
# Phase 3: Fine-Tuning (Optional)
# ============================================================================
print("\n" + "=" * 70)
print("PHASE 3: FINE-TUNING")
print("=" * 70)

base_model.trainable = True
fine_tune_at = len(base_model.layers) - 20

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"Fine-tuning from layer {fine_tune_at} onwards")

FINE_TUNE_EPOCHS = 10
history_finetune = model.fit(
    train_generator,
    epochs=FINE_TUNE_EPOCHS,
    initial_epoch=len(history.epoch),
    validation_data=valid_generator,
    callbacks=callbacks,
    verbose=1
)

print("\nFine-tuning completed!")

# ============================================================================
# Phase 4: Evaluation & Analysis
# ============================================================================
print("\n" + "=" * 70)
print("PHASE 4: EVALUATION & ANALYSIS")
print("=" * 70)

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)

print("\nTEST SET EVALUATION:")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Generate predictions
test_generator.reset()
y_pred = model.predict(test_generator, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

class_names_ordered = [class_names[i] for i in range(len(class_names))]

# Classification Report
print("\nCLASSIFICATION REPORT:")
print(classification_report(
    y_true, 
    y_pred_classes, 
    target_names=class_names_ordered,
    digits=4
))

# Training Curves
print("\nGenerating training curves...")
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

acc = history.history['accuracy'] + history_finetune.history['accuracy']
val_acc = history.history['val_accuracy'] + history_finetune.history['val_accuracy']
loss = history.history['loss'] + history_finetune.history['loss']
val_loss = history.history['val_loss'] + history_finetune.history['val_loss']
epochs_range = range(1, len(acc) + 1)

axes[0].plot(epochs_range, acc, 'b-', label='Training Accuracy', linewidth=2)
axes[0].plot(epochs_range, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].legend(loc='lower right')
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_range, loss, 'b-', label='Training Loss', linewidth=2)
axes[1].plot(epochs_range, val_loss, 'r-', label='Validation Loss', linewidth=2)
axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
print("Saved: training_curves.png")
plt.close()

# Confusion Matrix
print("\nGenerating confusion matrix...")
cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names_ordered,
            yticklabels=class_names_ordered,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
print("Saved: confusion_matrix.png")
plt.close()

# Error Analysis
misclassified_indices = np.where(y_pred_classes != y_true)[0]
test_generator.reset()
filepaths = test_generator.filepaths

if len(misclassified_indices) > 0:
    print("\nGenerating error analysis visualization...")
    num_to_show = min(5, len(misclassified_indices))
    fig, axes = plt.subplots(1, num_to_show, figsize=(20, 4))
    if num_to_show == 1:
        axes = [axes]
    fig.suptitle('Error Analysis: Misclassified Images', fontsize=16, fontweight='bold')
    
    for i, idx in enumerate(misclassified_indices[:num_to_show]):
        img_path = filepaths[idx]
        img = mpimg.imread(img_path)
        axes[i].imshow(img)
        true_class = class_names_ordered[y_true[idx]]
        pred_class = class_names_ordered[y_pred_classes[idx]]
        confidence = y_pred[idx][y_pred_classes[idx]]
        axes[i].set_title(f'True: {true_class}\nPred: {pred_class}\nConf: {confidence:.2f}', 
                          fontsize=10, fontweight='bold')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('error_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved: error_analysis.png")
    plt.close()
    
    print("\nERROR ANALYSIS COMMENTS:")
    num_to_analyze = min(5, len(misclassified_indices))
    for i, idx in enumerate(misclassified_indices[:num_to_analyze], 1):
        true_class = class_names_ordered[y_true[idx]]
        pred_class = class_names_ordered[y_pred_classes[idx]]
        confidence = y_pred[idx][y_pred_classes[idx]]
        print(f"\n{i}. True: {true_class} | Predicted: {pred_class} | Confidence: {confidence:.2f}")
else:
    print("\nNo misclassified images! Model achieved perfect accuracy on test set.")

# ============================================================================
# Phase 5: Deployment / Inference
# ============================================================================
print("\n" + "=" * 70)
print("PHASE 5: DEPLOYMENT / INFERENCE")
print("=" * 70)

def predict_fruit(image_path):
    """Predict fruit class from an image path."""
    from tensorflow.keras.preprocessing import image
    
    img = image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    predicted_class = class_names_ordered[predicted_class_idx]
    
    print("\nPREDICTION RESULTS:")
    print(f"Image: {image_path}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence Score: {confidence:.4f} ({confidence*100:.2f}%)")
    
    return predicted_class, confidence

# Test the function
print("\nTesting predict_fruit function...")
sample_image_path = list((TEST_DIR / class_names[0]).glob('*.jpg'))[0]
predicted_class, confidence = predict_fruit(str(sample_image_path))

print("\n" + "=" * 70)
print("ALL PHASES COMPLETED SUCCESSFULLY!")
print("=" * 70)
print("\nGenerated files:")
print("- sample_images.png")
print("- distribution_plot.png")
print("- training_curves.png")
print("- confusion_matrix.png")
print("- error_analysis.png (if errors found)")
print("- best_fruit_model.h5")
print("\nDone!")

