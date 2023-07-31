import fastcore
import fastai
from fastcore.all import *
from fastai.vision.all import *
import streamlit as st

import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

## LOAD MODEl
learn_inf = load_learner("export.pkl")
## CLASSIFIER
def classify_img(data):
    pred, pred_idx, probs = learn_inf.predict(data)
    return pred, probs[pred_idx]
## STREAMLIT
st.title("Classification of Ecoli")
bytes_data = None
uploaded_image = st.file_uploader("Choose your image:")
if uploaded_image:
    bytes_data = uploaded_image.getvalue()
    st.image(bytes_data, caption="Uploaded image")
if bytes_data:
    classify = st.button("Check!")
    if classify:
        label, confidence = classify_img(bytes_data)
        st.write(f"{label}!")
