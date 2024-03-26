
import streamlit as st
import tensorflow as tf
import numpy as np 
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

def load_model():
    return tf.keras.models.load_model('finalmodel.hdf5')

classes = {
    0: 'anger😠',
    1: 'contempt 😏',
    2:  'disgust 🤢',
    3: 'fear 😨',
    4: 'happy 😄',
    5:  'sadness 😢',
    6: 'surprise 😲'
}

model = load_model()
def display_logo():
    st.image("iiuc-logo.png", width=125)

def fixed_logo_layout():
 
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        display_logo()

fixed_logo_layout()
# Function to display logo

st.title("Exploring Facial Expression of Individuals Using ACNN")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], accept_multiple_files=False)

if uploaded_file is not None:
    img = Image.open(io.BytesIO(uploaded_file.read()))
    img1 = img.convert('RGB')
    img_resized = img1.resize((48, 48))  
    st.image(img, caption='Uploaded Image', width=128)  
    
    x = tf.keras.preprocessing.image.img_to_array(img_resized)
    x = np.expand_dims(x, axis=0)
    
    pred = model.predict(x)
    
    class_names = list(classes.values())
    pred_cls = class_names[np.argmax(pred)]
    pred_prob = np.max(pred)
    
    st.write("Predicted Class:", pred_cls)
    st.write("Probability:", 100 * pred_prob, "percent")
