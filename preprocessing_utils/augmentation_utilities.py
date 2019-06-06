# For python 2 support
from __future__ import absolute_import, print_function, unicode_literals, division

# import necessary packages
import tensorflow as tf
import numpy as np

# NOTE: Private Function
# TODO ==> Do private functions if needed here


# NOTE: Public Function

@tf.function
def crop(img, crop_min_scale_percent = 0.8, threshold = 0.5):
    """
    Function to let us do random cropping in range of 
    crop_min_scale_percent value till 1.0 scales. Threshold
    sets a chance to allow random cropping. eg. threshold = 0.5
    means only 50% of the time, we will do random cropping
    
    Arguments:
        img {tf.Tensor} -- The input image tensor
    
    Keyword Arguments:
        crop_min_scale_percent {float} -- The min scale percent for random cropping (default: {0.8})
        threshold {float} -- Threshold sets a chance to allow random cropping. 
        eg. threshold = 0.5 means only 50% of the time, we will do random 
        cropping (default: {0.5})
    
    Returns:
        tf.Tensor -- The random cropped image
    """
    scale = tf.random.uniform(shape = [], minval = crop_min_scale_percent, maxval = 1.0, dtype = tf.float32)
    scaled_shape = (tf.cast(256.0 * scale, dtype = tf.int32), tf.cast(128.0 * scale, dtype = tf.int32), 3)
    
    # lets do random crops only for the threshold we have
    choice = tf.random.uniform(shape = [], minval = 0.0, maxval = 1.0, dtype = tf.float32)
    
    return tf.cond(choice <= threshold, lambda: img, lambda: tf.image.random_crop(img, size = scaled_shape))

@tf.function
def rotate(img):
    """
    Function to do random 90 deg rotations
    to the image as augmentation
    
    Arguments:
        img {tf.Tensor} -- The image to augment
    
    Returns:
        tf.Tensor -- the augmented image
    """
    img = tf.image.rot90(img, tf.random.uniform(shape = [], minval = 0, maxval = 4, dtype = tf.int32))
    return img

@tf.function
def flip(img):
    """
    Function to do random left-right-up-down
    flipd
    
    Arguments:
        img {tf.Tensor} -- The image tensor to flip
    
    Returns:
        tf.Tensor -- The augmented image output
    """
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    return img

@tf.function
def color(img):
    """
    Function to do random color augmentation such
    as hue, saturation, brightness, contrast adjustment
    
    Arguments:
        img {tf.Tensor} -- The image tensor to flip
    
    Returns:
        tf.Tensor -- the augmented image
    """
    img = tf.image.random_hue(img, 0.08)
    img = tf.image.random_saturation(img, 0.6, 1.6)
    img = tf.image.random_brightness(img, 0.05)
    img = tf.image.random_contrast(img, 0.7, 1.3)
    return img