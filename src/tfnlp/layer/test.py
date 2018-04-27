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
            #self.check_reuse(scope)
            #if self.call_cnt ==0:
                #self.cell = LSTMCell(self.output_dim,initializer = self.initializer)
                #cell = BasicLSTMCell(self.output_dim)
            print scope.reuse
            rnn.dynamic_rnn(self.cell,inputs,seq_len,dtype = inputs.dtype)
            print scope.reuse
            return rnn.dynamic_rnn(self.cell,inputs,seq_len,dtype = inputs.dtype)
            
            #return rnn.static_rnn(self.cell,inputs.as_list(),dtype = inputs.dtype)
if __name__ =="__main__":

    a = tf.Variable([[[1.0,2],[3,4]],[[5,6],[7,8]]])
    b = tf.Variable([[[1.0,2],[3,4]],[[5,6],[7,8]]])
    cell1 = LSTMCell(200)
    #print cell1.scope_name
    with tf.variable_scope("aaa") as scope:    
        #print scope.name,scope.reuse
        nn = rnn.dynamic_rnn(cell1,a,[1,2],dtype = a.dtype)
        print cell1.scope_name 
        #print tf.trainable_variables()
        
        #print scope.name,scope.reuse
        nn = rnn.dynamic_rnn(cell1,b,[1,2],dtype = a.dtype)
        #print tf.trainable_variables()
        #with tf.variable_scope("rnn/lstm_cell") as s:
        #    print s.name,s.reuse
    #print tf.trainable_variables()
    #cell2 = LSTMCell(200)
    #with tf.variable_scope("aaa") as scope:
    #    nn = rnn.dynamic_rnn(cell2,a,[1,2],dtype = a.dtype)

    #
    #lstm = LSTMLayer("LSTM",10)
 
 
    #output = lstm(a,seq_len = [1,2])
    #output2 = lstm(a,seq_len = [1,2])
