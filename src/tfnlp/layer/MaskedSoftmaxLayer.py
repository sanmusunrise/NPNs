import tensorflow as tf
from tensorflow.python.ops import rnn
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import *
from tensorflow.python.util import nest
from Layer import Layer
from MaskLayer import *

class MaskedSoftmaxLayer(Layer):
    
    def __call__(self, inputs,seq_len = None):
        if inputs.dtype.is_integer:
            inputs = tf.cast(inputs,dtype = tf.float32)
        exp_val = tf.exp(inputs)
        if seq_len != None:
            with tf.variable_scope(self.scope) as scope:
                if self.call_cnt ==0:
                    self.mask = MaskLayer(scope = "Mask", reuse = self.reuse)
                self.check_reuse(scope)
                exp_val = self.mask(exp_val,seq_len)
        return exp_val / (self.epsilon + tf.reduce_sum(exp_val,axis = 1,keep_dims = True))
    
    def set_extra_parameters(self,paras =None):
        self.epsilon = 1e-8
        if paras and "epsilon" in paras:
            self.epsilon = paras["epsilon"]
        
if __name__ =="__main__":

    a = tf.Variable([[1,2,3],[4,5,6]])
    mask = MaskedSoftmaxLayer("maskSoftmax")
    output = mask(a,seq_len = [3,2])
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print sess.run(output)
