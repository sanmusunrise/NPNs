import tensorflow as tf
class Layer():
    
    def __init__(self,scope,output_dim = -1,reuse = None):
        self.scope = scope
        self.reuse = reuse
        self.output_dim = output_dim
        self.call_cnt = 0
        self.initializer = None
        
        self.set_initializer()
        self.set_extra_parameters()
        self.set_extra_feeds()
        
    def __call__(self,inputs,seq_len = None):
        pass
    
    def set_extra_parameters(self,parameters = None):
        pass
    def set_extra_feeds(self,feeds = None):
        pass
    
    def check_reuse(self,scope):
        if self.call_cnt >0:
            if self.reuse == True:
                scope.reuse_variables()
            if self.reuse == None:
                if self.call_cnt >0:
                    print "Warning: Reuse variable with reuse value = None in scope",scope.name
                    scope.reuse_variables()
            if self.reuse == False:
                if self.call_cnt >0:
                    print "Error: reuse variable with reuse value = False in scope",scope.name
                    exit(-1)
        self.call_cnt +=1
    
    def set_initializer(self,initializer = None):
        if not initializer:
            initializer = tf.contrib.layers.xavier_initializer
        self.initializer = initializer
