import tensorflow as tf
from tensorflow.python.ops import rnn
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import *
from tensorflow.python.util import nest
from Layer import Layer

class BatchNormalizationLayer(Layer):

    def __call__(self,x, seq_len = None):
        """
        Code taken from http://stackoverflow.com/a/34634291/2267819
        """
        n_out = int(x.get_shape()[-1])
        decay = self.decay
        eps = self.eps
        stddev = self.stddev
        phase_train = self.phase_train
        with tf.variable_scope(self.scope) as scope:
            self.check_reuse(scope)
            
            beta = tf.get_variable(name='beta', shape=[n_out], initializer=tf.constant_initializer(0.0)
                                   , trainable=True)
            gamma = tf.get_variable(name='gamma', shape=[n_out], initializer=tf.random_normal_initializer(1.0, stddev),
                                    trainable=True)
            batch_mean, batch_var = tf.nn.moments(x, self.normal_dim, name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=decay)
            
            def mean_var_with_update():
                with tf.variable_scope(tf.get_variable_scope(), reuse=None):
                    ema_apply_op = ema.apply([batch_mean, batch_var])
                    with tf.control_dependencies([ema_apply_op]):
                        return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(phase_train,
                                mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)
        return normed
        
    def set_extra_parameters(self,paras =None):
        self.decay = 0.9
        self.eps = 1e-5
        self.stddev = 0.02
        self.normal_dim = [0]
        self.reuse = False
        
        if not paras:
            return
        
        if "decay" in paras:
            self.decay = paras["decay"]
        if "eps" in paras:
            self.eps = paras["eps"]
        if "stddev" in paras:
            self.stddev = paras["stddev"]
        if "normal_dim" in paras:
            self.normal_dim = paras["normal_dim"]
            
    def set_extra_feeds(self,feeds = None):
        if feeds and "phase_train" in feeds:
            self.phase_train = feeds["phase_train"]
        else:
            self.phase_train = tf.Variable(True)
            
            
if __name__ =="__main__":

    a = tf.Variable([[[1.0,2],[3,4]],[[5,6],[7,8]]])
    bn = BatchNormalizationLayer("bn")
    output = bn(a)