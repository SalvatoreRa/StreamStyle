import streamlit as st
import tensorflow as tf 
import tensorflow_hub as hub
import numpy as np
import pd 

# https://en.wikipedia.org/wiki/Andor_(TV_series)

def load_images():
     content = st.file_uploader("Choose the image to paint!")
     style = st.file_uploader("Choose the style!")
    return content, style

def load_model():
    model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    return model

# Create the main app
def main():
    st.title("Neural Style Transfer")

    content, style = load_images()


if __name__ == "__main__":
    main()
