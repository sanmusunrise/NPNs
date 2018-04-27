import sys
sys.path.append('../')
import tensorflow as tf
from layer.Layer import Layer
from layer.MaskLayer import MaskLayer

class AverageSentenceEncoder(Layer):
    ''' 
    args: inputs - B*T*d vector
    '''
    def __call__(self,inputs,seq_len = None):
        with tf.variable_scope(self.scope) as scope:
            
            if self.call_cnt ==0:
                self.mask = MaskLayer("Mask", reuse = self.reuse)
            self.check_reuse(scope)
            summed_vec = tf.reduce_sum(self.mask(inputs,seq_len) ,axis = 1)
            return summed_vec / (tf.cast( tf.reshape(seq_len,[-1,1]), summed_vec.dtype) + self.epsilon)
        
    def set_extra_parameters(self,paras =None):
        self.epsilon = 1e-8
        if paras and "epsilon" in paras:
            self.epsilon = paras["epsilon"]
            
if __name__ =="__main__":

    a = tf.Variable([[[1.0,2],[3,4]],[[5,6],[7,8]]])
    b = tf.Variable([[[4.0,2],[3,4]],[[5,6],[7,8]]])
    mask = AverageSentenceEncoder("ASE")
    output = mask(a,seq_len = [2,1])
    output2 = mask(a,seq_len = [2,1])
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print sess.run([output,output2])
