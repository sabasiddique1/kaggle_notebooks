

# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))


# Set data paths
# For Google Colab: mount drive and set path accordingly
# For local: use relative paths
BASE_DIR = Path('../data')
TRAIN_DIR = BASE_DIR / 'train'
VALID_DIR = BASE_DIR / 'valid'
TEST_DIR = BASE_DIR / 'test'

# Identify Classes: List sub-folders in train directory
class_names = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
print("=" * 50)
print("CLASSES IDENTIFIED:")
print("=" * 50)
for i, class_name in enumerate(class_names, 1):
    print(f"{i}. {class_name}")
print(f"\nTotal number of classes: {len(class_names)}")


# Visualize Samples: Display one sample image for every class
import matplotlib.image as mpimg

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle('Sample Images from Each Fruit Class', fontsize=16, fontweight='bold')

for idx, class_name in enumerate(class_names):
    row = idx // 5
    col = idx % 5
    
    # Get first image from each class
    class_dir = TRAIN_DIR / class_name
    image_files = list(class_dir.glob('*.jpg'))
    
    if image_files:
        img_path = image_files[0]
        img = mpimg.imread(img_path)
        axes[row, col].imshow(img)
        axes[row, col].set_title(class_name.title(), fontsize=12, fontweight='bold')
        axes[row, col].axis('off')

plt.tight_layout()
plt.show()


# Distribution Plot: Number of images per class in train set
train_counts = {}
for class_name in class_names:
    class_dir = TRAIN_DIR / class_name
    train_counts[class_name] = len(list(class_dir.glob('*.jpg')))

# Create bar chart
plt.figure(figsize=(14, 6))
bars = plt.bar(train_counts.keys(), train_counts.values(), color='steelblue', edgecolor='black')
plt.title('Distribution of Images per Class in Training Set', fontsize=14, fontweight='bold')
plt.xlabel('Fruit Class', fontsize=12)
plt.ylabel('Number of Images', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}',
             ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

# Print summary statistics
print("\n" + "=" * 50)
print("TRAINING SET STATISTICS:")
print("=" * 50)
total_images = sum(train_counts.values())
print(f"Total images: {total_images}")
print(f"Average per class: {total_images / len(class_names):.1f}")
print(f"Min images: {min(train_counts.values())}")
print(f"Max images: {max(train_counts.values())}")


# Configuration
IMG_SIZE = 224  # Standard size for MobileNetV2, ResNet50, VGG16
BATCH_SIZE = 32
NUM_CLASSES = len(class_names)

# Data Augmentation for Training Set
# Apply augmentation only to training data to prevent overfitting
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to 0-1 range
    rotation_range=20,  # Random rotation up to 20 degrees
    width_shift_range=0.2,  # Random horizontal shift
    height_shift_range=0.2,  # Random vertical shift
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True,  # Random horizontal flip
    fill_mode='nearest'  # Fill mode for transformations
)

# No augmentation for validation and test sets
# Only rescale to normalize pixel values
valid_test_datagen = ImageDataGenerator(rescale=1./255)

print("Data augmentation configured successfully!")
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

print("=" * 50)
print("DATA GENERATORS CREATED:")
print("=" * 50)
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {valid_generator.samples}")
print(f"Test samples: {test_generator.samples}")
print(f"\nClass indices: {train_generator.class_indices}")


# Load pre-trained MobileNetV2 without top layers (include_top=False)
# This gives us the feature extraction base
base_model = MobileNetV2(
    weights='imagenet',  # Pre-trained on ImageNet
    include_top=False,  # Exclude the classification head
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze the base model layers so they are not updated during initial training
base_model.trainable = False

print("Base model (MobileNetV2) loaded and frozen!")
print(f"Number of layers in base model: {len(base_model.layers)}")
print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in base_model.trainable_weights])}")


# Build the complete model
# Add custom dense layers on top of the frozen base
model = models.Sequential([
    base_model,  # Frozen MobileNetV2 base
    layers.GlobalAveragePooling2D(),  # Global average pooling
    layers.Dense(512, activation='relu'),  # Dense layer with ReLU
    layers.Dropout(0.5),  # Dropout to reduce overfitting
    layers.Dense(256, activation='relu'),  # Another dense layer
    layers.Dropout(0.3),  # Another dropout layer
    layers.Dense(NUM_CLASSES, activation='softmax')  # Output layer with Softmax
])

# Display model architecture
print("=" * 50)
print("MODEL ARCHITECTURE:")
print("=" * 50)
model.summary()


# Compile the model
# Using Adam optimizer and CategoricalCrossentropy loss as required
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model compiled successfully!")
print("Optimizer: Adam")
print("Loss: Categorical Crossentropy")
print("Metrics: Accuracy")


# Define callbacks
# Early Stopping to prevent unnecessary computation
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,  # Stop if no improvement for 5 epochs
    restore_best_weights=True,
    verbose=1
)

# Model checkpoint to save best model
checkpoint = ModelCheckpoint(
    'best_fruit_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Reduce learning rate on plateau
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=0.00001,
    verbose=1
)

callbacks = [early_stopping, checkpoint, reduce_lr]

print("Callbacks configured:")
print("- Early Stopping (patience=5)")
print("- Model Checkpoint (saves best model)")
print("- Reduce LR on Plateau")


# Train the model
# Minimum 10 epochs as required
EPOCHS = 20  # Train for up to 20 epochs (early stopping will stop if needed)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=valid_generator,
    callbacks=callbacks,
    verbose=1
)

print("\nTraining completed!")


# Fine-tuning: Unfreeze some top layers
# Unfreeze the last 20 layers of the base model
base_model.trainable = True
fine_tune_at = len(base_model.layers) - 20

# Freeze all layers before fine_tune_at
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Recompile with a very low learning rate (1e-5 as suggested)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"Fine-tuning from layer {fine_tune_at} onwards")
print(f"Trainable layers: {sum([1 for layer in base_model.layers if layer.trainable])}")
print(f"Learning rate: 1e-5")


# Fine-tune the model
FINE_TUNE_EPOCHS = 10

history_finetune = model.fit(
    train_generator,
    epochs=FINE_TUNE_EPOCHS,
    initial_epoch=history.epoch[-1],
    validation_data=valid_generator,
    callbacks=callbacks,
    verbose=1
)

print("\nFine-tuning completed!")


# Evaluate on test set
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)

print("=" * 50)
print("TEST SET EVALUATION:")
print("=" * 50)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")


# Generate predictions for classification report
test_generator.reset()
y_pred = model.predict(test_generator, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

# Get class names in correct order
class_names_ordered = [class_names[i] for i in range(len(class_names))]

# Classification Report
print("=" * 50)
print("CLASSIFICATION REPORT:")
print("=" * 50)
print(classification_report(
    y_true, 
    y_pred_classes, 
    target_names=class_names_ordered,
    digits=4
))


# Training Curves: Plot Training vs Validation Accuracy and Loss
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Combine history from initial training and fine-tuning
if 'history_finetune' in globals():
    # Combine both histories
    acc = history.history['accuracy'] + history_finetune.history['accuracy']
    val_acc = history.history['val_accuracy'] + history_finetune.history['val_accuracy']
    loss = history.history['loss'] + history_finetune.history['loss']
    val_loss = history.history['val_loss'] + history_finetune.history['val_loss']
    epochs_range = range(1, len(acc) + 1)
else:
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(acc) + 1)

# Plot Accuracy
axes[0].plot(epochs_range, acc, 'b-', label='Training Accuracy', linewidth=2)
axes[0].plot(epochs_range, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].legend(loc='lower right')
axes[0].grid(True, alpha=0.3)

# Plot Loss
axes[1].plot(epochs_range, loss, 'b-', label='Training Loss', linewidth=2)
axes[1].plot(epochs_range, val_loss, 'r-', label='Validation Loss', linewidth=2)
axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# Confusion Matrix
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
plt.show()


# Find misclassified images
misclassified_indices = np.where(y_pred_classes != y_true)[0]

# Get file paths for misclassified images
test_generator.reset()
filepaths = test_generator.filepaths

# Display 5 misclassified images
if len(misclassified_indices) > 0:
    num_to_show = min(5, len(misclassified_indices))
    fig, axes = plt.subplots(1, num_to_show, figsize=(20, 4))
    if num_to_show == 1:
        axes = [axes]
    fig.suptitle('Error Analysis: Misclassified Images', fontsize=16, fontweight='bold')
    
    for i, idx in enumerate(misclassified_indices[:num_to_show]):
        # Get the image path and load it
        img_path = filepaths[idx]
        img = mpimg.imread(img_path)
        
        # Display image
        axes[i].imshow(img)
        true_class = class_names_ordered[y_true[idx]]
        pred_class = class_names_ordered[y_pred_classes[idx]]
        confidence = y_pred[idx][y_pred_classes[idx]]
        
        axes[i].set_title(f'True: {true_class}\nPred: {pred_class}\nConf: {confidence:.2f}', 
                          fontsize=10, fontweight='bold')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
else:
    print("No misclassified images found! Perfect accuracy!")

# Print error analysis comments
print("=" * 50)
print("ERROR ANALYSIS COMMENTS:")
print("=" * 50)
if len(misclassified_indices) > 0:
    num_to_analyze = min(5, len(misclassified_indices))
    for i, idx in enumerate(misclassified_indices[:num_to_analyze], 1):
        true_class = class_names_ordered[y_true[idx]]
        pred_class = class_names_ordered[y_pred_classes[idx]]
        confidence = y_pred[idx][y_pred_classes[idx]]
        
        print(f"\n{i}. True: {true_class} | Predicted: {pred_class} | Confidence: {confidence:.2f}")
        
        # Provide brief comment on why failure might have happened
        if 'apple' in true_class.lower() and 'orange' in pred_class.lower():
            print("   Comment: Similar round shape and color may have caused confusion.")
        elif 'banana' in true_class.lower() and 'mango' in pred_class.lower():
            print("   Comment: Similar elongated shape and yellow color may have caused confusion.")
        elif 'cherry' in true_class.lower() and 'strawberries' in pred_class.lower():
            print("   Comment: Similar small red fruit appearance may have caused confusion.")
        else:
            print(f"   Comment: Visual similarity in shape, color, or texture between {true_class} and {pred_class}.")
else:
    print("\nNo misclassified images! Model achieved perfect accuracy on test set.")


def predict_fruit(image_path):
    """
    Predict fruit class from an image path.
    
    Args:
        image_path: Path to the image file (local path)
    
    Returns:
        predicted_class: The predicted fruit class name
        confidence: Confidence score (probability) for the prediction
    """
    from tensorflow.keras.preprocessing import image
    
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    predicted_class = class_names_ordered[predicted_class_idx]
    
    # Print results
    print("=" * 50)
    print("PREDICTION RESULTS:")
    print("=" * 50)
    print(f"Image: {image_path}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence Score: {confidence:.4f} ({confidence*100:.2f}%)")
    print("=" * 50)
    
    return predicted_class, confidence

# Test the function with a sample image
print("Testing predict_fruit function with a sample image...")
# Get a sample image from test set
sample_image_path = list((TEST_DIR / class_names[0]).glob('*.jpg'))[0]
predicted_class, confidence = predict_fruit(str(sample_image_path))
