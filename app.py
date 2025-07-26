import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page setup
st.set_page_config(page_title="Fake vs Real Face Detector", layout="centered")

# Load model
model = tf.keras.models.load_model("model/fake_image_model.h5")
class_names = ['fake', 'real']

# Title
st.title("🧠 Fake vs Real Face Detector")
st.write("Upload a face image to check if it's real or AI-generated.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)

    # Layout: Image on left, Result on right
    col1, col2 = st.columns([1, 1])  # 50-50 split

    with col1:
        st.image(image, caption="Uploaded Image", width=300)

    with col2:
    # Preprocess
        img = image.resize((224, 224))
        img_array = tf.keras.utils.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)[0][0]
        label = class_names[int(prediction > 0.5)]
        confidence = prediction if prediction > 0.5 else 1 - prediction
        confidence_pct = f"{confidence * 100:.2f}%"

        # Result section
        st.subheader("📋 Prediction Result")

        if label == "fake":
            st.error(" **Prediction: FAKE IMAGE**")
            #st.write(f"🔍 **Confidence:** {confidence_pct}")
            st.markdown("""
            This image appears to be **AI-generated** (possibly created using GANs or tools like ThisPersonDoesNotExist).
            
            **Why does this matter?**
            - Fake faces can be used in fake identities, fraud, or disinformation.
            - They're highly realistic and often hard to distinguish with the human eye.

            >  Always verify images from unknown or untrusted sources.
            """)
        else:
            st.success("**Prediction: REAL IMAGE**")
            #st.write(f"🔍 **Confidence:** {confidence_pct}")
            st.markdown("""
            This image appears to be a **real human face**, most likely taken from a camera or device.

            **What it means:**
            - The face is not detected as synthetic.

            > Best used for basic detection & awareness purposes.
            """)

else:
    st.info("Please upload a face image to get a prediction.")
