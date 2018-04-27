import tensorflow as tf
from Layer import Layer

class Conv1DLayer(Layer):
    
    def __call__(self,inputs,seq_len = None):
        with tf.variable_scope(self.scope) as scope:
            
            if self.call_cnt ==0:
                inputs_shape = inputs.get_shape().as_list()
                filter_shape = [self.window,inputs_shape[2],self.output_dim]

                self.filters = tf.get_variable("filters",filter_shape,initializer = self.initializer(dtype = inputs.dtype) )
                if self.use_bias:
                    self.bias = tf.get_variable("bias",[self.output_dim],initializer = self.initializer(dtype = inputs.dtype))
                else:
                    self.bias = 0
                    
            self.check_reuse(scope)
            return tf.nn.conv1d(inputs,self.filters,1,self.padding_mode,name = "conv1d") + self.bias
    
    def set_extra_parameters(self,parameters = None):
        self.window = 3
        self.use_bias = True
        self.padding_mode = "SAME"
        if not parameters:
            return 
        if 'window' in parameters:
            self.window = parameters['window']
        if 'use_bias' in parameters:
            self.use_bias = parameters['use_bias']
if __name__ =="__main__":

    inputs = tf.placeholder(dtype = tf.float32,shape = [None,3,3])
    a = [[[1,2,3],[4,5,6],[7,8,9]],[[7,8,9],[10,11,12],[1,2,3]]]

    layer = Conv1DLayer("conv1d",3)
    layer.set_initializer(tf.ones_initializer)
    
    output = layer(inputs)
    #output2 = layer(inputs)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    print sess.run(output,feed_dict={inputs:a})

    for var in tf.trainable_variables():
        print var.name
