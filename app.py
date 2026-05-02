import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("mask_model.keras")

def predict_mask(image):
    # Convert to RGB (important)
    img = image.convert("RGB")

    # Resize
    img = img.resize((150, 150))

    # Normalize
    img_array = np.array(img) / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    prob = prediction[0][0]

    # Result
    if prob > 0.5:
        return {"Without Mask ❌": float(prob), "With Mask 😷": float(1-prob)}
    else:
        return {"With Mask 😷": float(1-prob), "Without Mask ❌": float(prob)}

# Interface
demo = gr.Interface(
    fn=predict_mask,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="Face Mask Detection",
    description="Upload an image to check if person is wearing a mask"
)

demo.launch()
