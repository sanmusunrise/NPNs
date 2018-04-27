import tensorflow as tf
from tensorflow.python.ops import rnn
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import *
from tensorflow.python.util import nest
from Layer import Layer

class MaskLayer(Layer):

    def __call__(self, m,seq_len):
        with tf.variable_scope(self.scope) as scope:
            self.check_reuse(scope)
            
            max_length = int(m.get_shape()[1])
            seq_len_mask = tf.sequence_mask(seq_len,maxlen = max_length, dtype = m.dtype)
            rank = m.get_shape().ndims
            extra_ones = tf.ones(rank - 2, dtype=tf.int32)
            seq_len_mask = tf.reshape(seq_len_mask, tf.concat((tf.shape(seq_len_mask), extra_ones), 0))
            return m * seq_len_mask
        
if __name__ =="__main__":

    a = tf.Variable([[[1.0,2],[3,4]],[[5,6],[7,8]]])
    mask = MaskLayer("mask")
    output = mask(a,seq_len = [1,2])
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print sess.run(output)
