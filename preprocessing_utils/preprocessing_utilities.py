# For python 2 support
from __future__ import absolute_import, print_function, unicode_literals, division

# import necessary packages
import tensorflow as tf
import numpy as np

# Import in-house packages
import preprocessing_utils.augmentation_utilities as augmentation_utilities

# NOTE: Private Function
# TODO ==> Insert any private functions here

# NOTE: Public Functions
@tf.function
def min_max_scaling(img, old_range:tuple, new_range:tuple):
    """
    Function to min max scaling for the given image from
    the old range to the new range given as tuple
    
    Arguments:
        img {tf.Tensor} -- The image to rescale
        old_range {tuple} -- The old range of scale values eg. (0., 255.)
        new_range {tuple} -- The new range of scale values to rescale eg. (0., 1.)
    
    Returns:
        tf.Tensor -- The output rescaled image
    """
    old_min, old_max = old_range
    new_min, new_max = new_range
    img = tf.add(tf.multiply(tf.divide(tf.subtract(img, old_min), tf.subtract(old_max, old_min)), tf.subtract(new_max, new_min)), new_min)
    return img

@tf.function
def image_read_and_augment(img_path: str, output_size: tuple, flip = False, color = False, crop = False, rotate = False):
    """
    Function to read the image JPEG from the image path and 
    do augmentation based on the options toggled
    
    Arguments:
        img_path {str} -- The path to the image to read
        output_size {tuple} -- The output size needed from the image loaded
    
    Keyword Arguments:
        flip {bool} -- The option to do flip augmentation (default: {False})
        color {bool} -- The option to do color augmentation (default: {False})
        crop {bool} -- The option to do crop augmentation (default: {False})
        rotate {bool} -- The option to do the rotate 90 deg. augmentation (default: {False})
    
    Returns:
        tf.Tensor -- The image tensor after decode and augmentation
    """
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels = 3)
    
    # Augment the image
    if crop:
        img = augmentation_utilities.crop(img, crop_min_scale_percent = 0.75, threshold = 0.5)
    
    if flip:
        img = augmentation_utilities.flip(img)
    
    if color:
        img = augmentation_utilities.color(img)
    
    if rotate:
        img = augmentation_utilities.rotate(img)
    
    # Resize the image
    d_w, d_h = output_size
    img = tf.image.resize(img, size = (d_w, d_h), method = tf.image.ResizeMethod.AREA)
    
    # Normalize the image
    img = min_max_scaling(img, (tf.reduce_min(img), tf.reduce_max(img)), (0.0, 1.0))

    return img

@tf.function
def preprocess_data(sample1_path, sample2_path, label) -> tuple:
    """
    Function to preprocess the images in sample1_path and
    sample2_path and apply basic augmentation techniques
    and also convert label into a one-hot encoder of 
    depth = 2 for the given MARS dataset model
    
    Arguments:
        sample1_path {tf.string} -- The sample 1 image path
        sample2_path {tf.string} -- The sample 2 image path
        label {tf.int32} -- The label
    
    Returns:
        tuple -- The (image 1, image 2), y tuple
    """
    # Read images
    img1 = image_read_and_augment(sample1_path, output_size = (299, 299), flip = True, color = True, crop = True, rotate = True)
    img2 = image_read_and_augment(sample2_path, output_size = (299, 299), flip = True, color = True, crop = True, rotate = True)

    label = tf.one_hot(label, depth = 2, dtype = tf.float32)

    return (img1, img2), label

def load_tf_dataset_generator(dataset_meta_path: str, batch_size = 32, shuffle = True, shuffle_buffer = 10000):
    """
    Function to return a tf.data.Dataset for the
    given dataset_metadata prepared from the 
    MARS dataset
    
    Arguments:
        dataset_meta_path {str} -- The path to the csv file
    
    Keyword Arguments:
        batch_size {int} -- The batch_size needed (default: {32})
        shuffle {bool} -- The shuffle b
        shuffle_buffer {int} -- The shuffle buffer for shuffling (default: {10000})
    
    Returns:
        tf.data.Dataset -- The generator for the dataset
    """

    # Get the dataset generator
    dataset_gen = tf.data.experimental.CsvDataset(dataset_meta_path, [tf.string, tf.string, tf.int32], header = True)

    # Map the preprocessor
    dataset_gen = dataset_gen.map(preprocess_data, num_parallel_calls = tf.data.experimental.AUTOTUNE)

    # Shuffle and repeat
    if shuffle:
        dataset_gen = dataset_gen.shuffle(shuffle_buffer)
    
    dataset_gen = dataset_gen.repeat()
    
    # Batch it
    dataset_gen = dataset_gen.batch(batch_size)

    # prefetch enabling
    dataset_gen = dataset_gen.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset_gen
    