import tensorflow as tf
from matplotlib import pyplot as plt
import preprocessing_utils
 
test_data = preprocessing_utils.load_tf_dataset_generator("./dataset/val_data.csv", shuffle_buffer = 500)

with tf.compat.v1.Session() as sess:
    data = sess.run(test_data)
    print(data)
# for elements in test_data:
#     print(elements)
#     break
