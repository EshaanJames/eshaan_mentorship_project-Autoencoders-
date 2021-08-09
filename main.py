import numpy as np
import streamlit as st
from PIL import Image
from keras.preprocessing.image import img_to_array
from skimage.color import lab2rgb
from skimage.transform import resize
from neural_network import model

st.title("Welcome to Image coloring App.")
st.sidebar.title("Select Inputs here:")
st.sidebar.subheader("Either upload image or select preloaded image")
image_selection = st.sidebar.selectbox("Select relevant option",
                                       options=["Default", "Upload your image"])
img1_color = []
if image_selection == "Upload your image":
    st.title("Upload + Classification Example")
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png', 'pdf', 'jpeg', "jfif"])
    if uploaded_file is not None:
        uploaded_file = Image.open(uploaded_file)
        img1 = img_to_array(uploaded_file)
        img1 = resize(img1, (256, 256))
        st.image((img1 / 256))
        img1_color.append(img1)
        img1_color = np.array(img1_color)

if st.sidebar.button("Show coloured image"):
    img1_color = img1_color.reshape(img1_color.shape+(1,))
    output1 = model.predict(img1_color)
    output1 = output1 * 128
    result = np.zeros((256, 256, 3))
    st.write(result.shape)
    st.write(img1_color.shape)
    st.write(output1[0].shape)
    result[:, :, 0] = img1_color[0][:, :, 0]
    result[:, :, 1:] = output1[0]
    r_img = lab2rgb(result)
    st.image(r_img)
