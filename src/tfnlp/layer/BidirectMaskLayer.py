import tensorflow as tf
from tensorflow.python.ops import rnn
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import *
from tensorflow.python.util import nest
from Layer import Layer
from MaskLayer import MaskLayer

class BidirectMaskLayer(Layer):
    
    '''
    The value out of [left_idx,right_idx) will be masked.
    '''
    def __call__(self, m,left_idx,right_idx):
        with tf.variable_scope(self.scope) as scope:
            if self.call_cnt ==0:
                self.left_mask = MaskLayer("left_mask")
                self.right_mask = MaskLayer("right_mask")
                self.left_mask.set_extra_parameters({"mask_from_right":False,"mask_value":self.mask_value})
                self.right_mask.set_extra_parameters({"mask_from_right":True,"mask_value":self.mask_value})
            self.check_reuse(scope)
            
            return self.right_mask( self.left_mask(m,left_idx) ,right_idx)
    
    def set_extra_parameters(self,parameters = None):
        self.mask_value = 0     
        if not parameters:
            return
        if "mask_value" in parameters:
            self.mask_value = parameters["mask_value"]
    
if __name__ =="__main__":

    a = tf.Variable([[[1.0,2],[3,4],[5,6]],[[5,6],[7,8],[9,10]]])
    mask = BidirectMaskLayer("mask")
    mask.set_extra_parameters({"mask_value":-100})
    output = mask(a,left_idx = [10,1],right_idx = [1,2])
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print sess.run(output)
