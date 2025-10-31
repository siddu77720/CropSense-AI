import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import os

def create_demo_model():
    """
    Create a demo disease detection model using MobileNetV2
    This is a placeholder - in real scenario, you'd train with actual data
    """
    print("🔄 Creating demo disease detection model...")
    
    # Define class names based on our disease database
    class_names = [
        "Healthy", "Early Blight", "Late Blight", "Powdery Mildew",
        "Leaf Spot", "Bacterial Spot", "Mosaic Virus", "Root Rot"
    ]
    
    # Load MobileNetV2 base model
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom classification head
    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(len(class_names), activation='softmax')(x)
    
    # Create model
    model = Model(inputs, outputs)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ Demo model created successfully!")
    
    # Create dummy training data for demonstration
    print("🔄 Generating demo training data...")
    
    # Create dummy data (in real scenario, use actual images)
    num_samples = 100
    dummy_images = np.random.random((num_samples, 224, 224, 3))
    dummy_labels = np.random.randint(0, len(class_names), num_samples)
    dummy_labels = tf.keras.utils.to_categorical(dummy_labels, len(class_names))
    
    # Train for a few epochs (demo only)
    print("🔄 Training demo model (this is simulated)...")
    history = model.fit(
        dummy_images, dummy_labels,
        epochs=5,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Save the model
    model.save('crop_disease_model.h5')
    print("✅ Model saved as 'crop_disease_model.h5'")
    
    # Save class names
    with open('class_names.txt', 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    
    print("✅ Class names saved to 'class_names.txt'")
    print("🎉 Demo training completed! You can now run the Streamlit app.")
    
    return model

if __name__ == "__main__":
    print("🌱 Crop Disease Detection Model Training")
    print("=" * 50)
    
    # Check if TensorFlow is properly installed
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Keras Version: {tf.keras.__version__}")
    
    # Create and train demo model
    model = create_demo_model()
    
    # Display model summary
    print("\n📊 Model Architecture Summary:")
    print("=" * 30)
    model.summary()
    
    print("\n🚀 Next steps:")
    print("1. Run 'streamlit run app.py' to start the web app")
    print("2. Upload plant images for disease detection")
    print("3. For better accuracy, train with real plant disease dataset")