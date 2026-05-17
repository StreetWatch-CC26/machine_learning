import cv2
import numpy as np
import tensorflow as tf

def predict_road(image_path, model_path='road_validator.keras', threshold=0.5462523102760315):
    """
    Predict if image contains a road
    
    Args:
        image_path: path to image file
        model_path: path to .keras model file
        threshold: cutoff probability (default 0.447)
    
    Returns:
        (is_road: bool, probability: float)
    """
    # Load model (only once if you want to optimize)
    model = tf.keras.models.load_model(model_path)
    
    # Load and preprocess image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img_batch = np.expand_dims(img, axis=0)
    
    # Predict
    prob = model.predict(img_batch, verbose=0)[0][0]
    is_road = prob >= threshold
    
    return is_road, float(prob)

# Example usage
if __name__ == "__main__":
    is_road, prob = predict_road("/kaggle/input/datasets/dzakiyahakimaadila/banana-haha/Banana-Single.jpg")
    print(f"Is road: {is_road}, Confidence: {prob:.3f}")