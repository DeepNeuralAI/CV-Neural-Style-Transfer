import tensorflow as tf

content_layers = ['block5_conv2']

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)

def style_loss(style_outputs, style_targets):
  loss = 0
  for name in style_outputs.keys():
    loss += tf.reduce_mean((style_outputs[name]-style_targets[name])**2)
  return loss

def content_loss(content_outputs, content_targets):
  loss = 0
  for name in content_outputs.keys():
    loss += tf.reduce_mean((content_outputs[name]-content_targets[name])**2)
  return loss

def total_loss(outputs, targets, weights):
  style_outputs = outputs['style']
  content_outputs = outputs['content']
   
  style_targets = targets["style"]
  content_targets = targets["content"]

  style_weight = weights["style"]
  content_weight = weights["content"]
  
  sty_loss = style_loss(style_outputs, style_targets)
  sty_loss *= style_weight / num_style_layers

  cont_loss = content_loss(content_outputs, content_targets)
  cont_loss *= content_weight / num_content_layers
  
  loss = sty_loss + cont_loss
  return loss
