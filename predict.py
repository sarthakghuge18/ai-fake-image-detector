import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sys

# Load model
model = tf.keras.models.load_model("model/fake_image_model.h5")

# Class labels (based on folder structure: fake, real)
class_names = ['fake', 'real']

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # make batch of 1

    prediction = model.predict(img_array)[0][0]
    label = class_names[int(prediction > 0.5)]
    confidence = prediction if prediction > 0.5 else 1 - prediction

    print(f"🖼 Image: {img_path}")
    print(f"🔍 Prediction: {label} ({confidence*100:.2f}%)")

# Run from command line
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py path_to_image.jpg")
    else:
        predict_image(sys.argv[1])
