# example.py - Contoh lengkap cara pake

import tensorflow as tf
from src.preprocessing import preprocess

def main():
    # Load model
    model = tf.keras.models.load_model('model/road_validator.keras')
    
    # Load threshold
    with open('model/threshold.txt', 'r') as f:
        threshold = float(f.read())
    
    # Test dengan gambar (ganti pathnya)
    img_path = "test.jpg"
    img = preprocess(img_path)
    prob = model.predict(img)[0][0]
    is_road = prob >= threshold
    
    print(f"Image: {img_path}")
    print(f"Probability: {prob:.3f}")
    print(f"Is road: {is_road}")

if __name__ == "__main__":
    main()