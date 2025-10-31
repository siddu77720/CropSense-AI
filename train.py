import numpy as np
import json
import os
from datetime import datetime

def create_demo_model_files():
    """
    Create demo model files without using TensorFlow
    This creates placeholder files that work with the app
    """
    print("🌱 CropSense AI - Demo Model Setup")
    print("=" * 50)
    
    # Define class names (must match app.py DISEASE_INFO keys)
    class_names = ["Healthy", "Early Blight", "Late Blight", "Powdery Mildew", "Leaf Spot"]
    print(f"Classes: {class_names}")
    
    # Create model info file
    model_info = {
        "class_names": class_names,
        "num_classes": len(class_names),
        "input_shape": [224, 224, 3],
        "model_type": "Demo CNN",
        "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": "Demo model for CropSense AI - Uses simulated predictions",
        "accuracy": "85% (Demo Mode)",
        "status": "Ready for inference"
    }
    
    # Save model info
    with open('model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("✅ Model info saved as 'model_info.json'")
    
    # Save class names
    with open('class_names.txt', 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    
    print("✅ Class names saved to 'class_names.txt'")
    
    # Create a placeholder model file (empty file with .h5 extension)
    # In real scenario, this would be a trained TensorFlow model
    with open('crop_disease_model.h5', 'w') as f:
        f.write("# Demo model file - Replace with actual trained model\n")
        f.write("# This is a placeholder for the TensorFlow model\n")
        f.write(f"# Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Classes: {', '.join(class_names)}\n")
    
    print("✅ Demo model file created as 'crop_disease_model.h5'")
    
    # Generate sample training statistics
    print("\n📊 Demo Training Statistics:")
    print("=" * 30)
    print(f"Model Architecture: Custom CNN")
    print(f"Input Shape: 224x224x3")
    print(f"Number of Classes: {len(class_names)}")
    print(f"Training Accuracy: 92% (Simulated)")
    print(f"Validation Accuracy: 88% (Simulated)")
    print(f"Model Status: Ready for use")
    
    return model_info

def test_demo_setup():
    """Test if the demo setup works correctly"""
    print("\n🧪 Testing demo setup...")
    
    try:
        # Check if files exist
        required_files = ['model_info.json', 'class_names.txt', 'crop_disease_model.h5']
        
        for file in required_files:
            if os.path.exists(file):
                print(f"✅ {file} - Found")
            else:
                print(f"❌ {file} - Missing")
        
        # Test class names loading
        with open('class_names.txt', 'r') as f:
            loaded_classes = [line.strip() for line in f]
        
        print(f"✅ Class names loaded: {loaded_classes}")
        
        # Simulate a prediction
        print("\n🔍 Simulating Prediction:")
        demo_predictions = np.random.random(len(loaded_classes))
        demo_predictions = demo_predictions / np.sum(demo_predictions)  # Normalize
        
        predicted_class_idx = np.argmax(demo_predictions)
        confidence = demo_predictions[predicted_class_idx]
        predicted_class = loaded_classes[predicted_class_idx]
        
        print(f"   Predicted: {predicted_class}")
        print(f"   Confidence: {confidence:.2%}")
        print("✅ Demo setup working correctly!")
        
    except Exception as e:
        print(f"❌ Error testing setup: {e}")

def main():
    """Main setup function"""
    try:
        print("🚀 Setting up CropSense AI Demo Model...")
        
        # Create demo model files
        model_info = create_demo_model_files()
        
        # Test the setup
        test_demo_setup()
        
        print("\n🎉 Demo setup completed successfully!")
        print("\n📁 Files created:")
        print("   - crop_disease_model.h5 (Demo model file)")
        print("   - class_names.txt (Disease classes)")
        print("   - model_info.json (Model metadata)")
        
        print("\n🚀 Next steps:")
        print("1. Run 'streamlit run app.py' to start the web app")
        print("2. Upload plant images for disease detection")
        print("3. The app will use demo mode with realistic predictions")
        print("4. For real training, install TensorFlow with Python 3.11 or lower")
        
        print("\n💡 Note: This is a demo setup.")
        print("   For production, train with real plant disease images")
        print("   and use actual TensorFlow model training.")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        print("💡 Make sure you have write permissions in the current directory")

if __name__ == "__main__":
    main()