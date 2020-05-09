import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO


import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False

from img_utils import *
from style_utils import *
from style_content_model import *

import pdb

st.title("Neural Style Transfer")

st.sidebar.title('Features')
content_img_buffer = st.sidebar.file_uploader("Choose a Content Image", type=["png", "jpg", "jpeg"])
style_img_buffer = st.sidebar.file_uploader("Choose a Style Image", type=["png", "jpg", "jpeg"])

content_image = load_img(content_img_buffer)
style_image = load_img(style_img_buffer)

generated_image = st.empty()

extractor = StyleContentModel(style_layers, content_layers)

targets = {
    "style": extractor(style_image)['style'],
    "content": extractor(content_image)['content']
}

weights = {
    "style": 1e-2,
    "content": 1e4
}

image = tf.Variable(content_image)

opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

@tf.function()
def train_step(image):
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = total_loss(outputs, targets, weights)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))


import time
start = time.time()

epochs = 10
steps_per_epoch = 100

step = 0
for n in range(epochs):
  for m in range(steps_per_epoch):
    step += 1
    train_step(image)
    generated_image.image(tensor_to_image(image))
    print(".", end='')
  # print("Train step: {}".format(step))

end = time.time()
print("Total time: {:.1f}".format(end-start))
