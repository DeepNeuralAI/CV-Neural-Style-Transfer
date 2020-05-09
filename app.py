import streamlit as st
from streamlit import caching
import numpy as np
import tensorflow as tf
from PIL import Image

from img_utils import *
from style_utils import *
from style_content_model import *


st.title("Neural Style Transfer")
st.markdown('#### Inspiration: [A Neural Algorithm of Artistic Style - Gatys et al.](https://arxiv.org/abs/1508.06576)')
st.sidebar.title('Features')

# Features
learning_rate = st.sidebar.slider('Learning rate', 0.0, 0.1, .02)
epochs = st.sidebar.slider('Number of Epochs', 1, 100, 10)
steps_per_epoch = st.sidebar.slider('Steps Per Epoch', 1, 500, 100)

st.markdown(f'<br>', unsafe_allow_html=True)

content_img_buffer = st.file_uploader("Choose a Content Image", type=["png", "jpg", "jpeg"])
style_img_buffer = st.file_uploader("Choose a Style Image", type=["png", "jpg", "jpeg"])


if content_img_buffer:
  content_image = load_img(content_img_buffer)
  image = tf.Variable(content_image)


if style_img_buffer:
  style_image = load_img(style_img_buffer)

if content_img_buffer and style_img_buffer:
  st.sidebar.image([content_img_buffer, style_img_buffer], caption=['Content Image', 'Style Image'], use_column_width = True)

@tf.function()
def train_step(image):
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = total_loss(outputs, targets, weights)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))

@st.cache(suppress_st_warning = True)
def run_style_transfer(image, epochs, steps_per_epoch):
  generated_image.clear()
  step = 0
  for n in range(epochs):
    for m in range(steps_per_epoch):
      step += 1
      train_step(image)
      generated_image.image(tensor_to_image(image), caption = 'Generated Image', use_column_width = True)


if content_img_buffer and style_img_buffer:
  clicked = st.sidebar.button('Generate')
  if clicked:
    caching.clear_cache()
    extractor = StyleContentModel(style_layers, content_layers)
    opt = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.99, epsilon=1e-1)

    targets = {
      "style": extractor(style_image)['style'],
      "content": extractor(content_image)['content']
    }

    weights = {
      "style": 1e-2,
      "content": 1e4
    }

    generated_image = st.empty()
    run_style_transfer(image, epochs, steps_per_epoch)
