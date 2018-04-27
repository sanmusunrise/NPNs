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
            if not self.mask_from_right:
                seq_len_mask = 1-seq_len_mask
            return m * seq_len_mask - ((seq_len_mask - 1) * self.mask_value)
    
    def set_extra_parameters(self,parameters = None):
        self.mask_value = 0
        self.mask_from_right = True
            
        if not parameters:
            return
        if "mask_value" in parameters:
            self.mask_value = parameters["mask_value"]
        if "mask_from_right" in parameters:
            self.mask_from_right = parameters["mask_from_right"]
    
if __name__ =="__main__":

    a = tf.Variable([[[1.0,2],[3,4]],[[5,6],[7,8]]])
    mask = MaskLayer("mask")
    seq_len = tf.placeholder(tf.int32,[None])
    params = {"mask_from_right":False,"mask_value":-100}
    mask.set_extra_parameters(params)
    output = mask(a,seq_len)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print sess.run(output,feed_dict ={seq_len:[1,3]})
    print sess.run(output,feed_dict ={seq_len:[1,1]})
