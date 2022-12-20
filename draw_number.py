import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2

import tensorflow as tf
from tensorflow import keras 
import numpy as np
# Specify canvas parameters in application
def draw():
    drawing_mode = st.sidebar.selectbox("Drawing tool:", ("freedraw", "point", "line", "rect", "circle", "transform") )
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 12)
    if drawing_mode == 'point':
        point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ", "#eee")
    bg_color = st.sidebar.color_picker("Background color hex: ")
    bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])

    realtime_update = st.sidebar.checkbox("Update in realtime", True)

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=realtime_update,
        height=250,
        width=275,
        drawing_mode=drawing_mode,
        point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
        key="canvas",
    )

    # Do something interesting with the image data and paths
    if canvas_result.image_data is not None:
        st.image(canvas_result.image_data)
        cv2.imwrite("digit.jpg", canvas_result.image_data)
