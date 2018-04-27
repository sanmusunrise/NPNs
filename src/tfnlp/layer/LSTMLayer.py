import tensorflow as tf
from Layer import Layer
from tensorflow.python.ops import rnn
from tensorflow.contrib.rnn import *
from tensorflow.python.util import nest

class LSTMLayer(Layer):

    def __call__(self,inputs,seq_len = None):
        if self.call_cnt ==0:
            self.cell = LSTMCell(self.output_dim,initializer = self.initializer(dtype=inputs.dtype))
        
        with tf.variable_scope(self.scope) as scope:
            self.check_reuse(scope)
            #if self.call_cnt ==0:
                #self.cell = LSTMCell(self.output_dim,initializer = self.initializer)
                #cell = BasicLSTMCell(self.output_dim)
            return rnn.dynamic_rnn(self.cell,inputs,seq_len,dtype = inputs.dtype)
            
            #return rnn.static_rnn(self.cell,inputs.as_list(),dtype = inputs.dtype)
if __name__ =="__main__":

    a = tf.Variable([[[1.0,2],[3,4]],[[5,6],[7,8]]])
    lstm = LSTMLayer("LSTM",10)
    output = lstm(a,seq_len = [1,2])
    #output2 = lstm(a,seq_len = [1,2])
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    output,state =  sess.run(output)
    output = output.tolist()
    for v in output:
        print v
    print 
    print state[1]
