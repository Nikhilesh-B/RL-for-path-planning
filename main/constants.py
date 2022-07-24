import tensorflow as tf

OBSTACLES = tf.reshape(tf.convert_to_tensor(()), (0, 3))
START = tf.constant([1 ,0])
GOAL = tf.constant([2.16, 3.36])
LINK_LENGTH = tf.constant([2, 2])