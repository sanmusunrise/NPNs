import tensorflow as tf
from tfnlp.embedding.GloveEmbeddings import *
from tfnlp.layer.Conv1DLayer import *
from tfnlp.layer.NegativeMaskLayer import *
from tfnlp.layer.MaskLayer import *
from EEData import *
from tfnlp.layer.DenseLayer import *
from tfnlp.layer.MaskedSoftmaxLayer import *
from scorer.event_scorer import *
import numpy as np
import shelve
from Configure import *
import sys


class NPN:
    def __init__(self, conf=None, model_save_path=None):
        self.configure = conf

        self.char_embeddings_file = None
        self.word_embeddings_file = None

        self.max_char_len = None
        self.max_word_len = None
        self.char_win_size = None
        self.word_win_size = None

        self.reg_label_size = None
        self.cls_label_size = None

        self.position_embedding_dim = None
        self.char_embedding_dim = None
        self.word_embedding_dim = None

        self.char_feature_map_size = None
        self.word_feature_map_size = None
        self.gate_vec_size = None

        self.model_save_path = None if conf else model_save_path
        self.result_output_path = None
        self.batch_size = None

        self.train_data = None
        self.test_data = None
        self.dev_data = None
        self.word2cnt = None
        self.char2cnt = None
        self.word_embeddings = None
        self.char_embeddings = None

        self.placeholders = {}
        self.variables = {}
        self.sess = tf.Session()

        self.epoch = 0
        self.best_dev_f1 = 0.0
        self.best_dev_epoch = 0

        self.set_configure() if self.configure else None

    def set_configure(self):
        for key in self.configure.confs:
            self.__dict__[key] = self.configure[key]

    def save_parameters(self):
        f = shelve.open(self.model_save_path + "model_obj.shv")
        f['configure'] = self.configure
        f['best_dev_f1'] = self.best_dev_f1
        f['best_dev_epoch'] = self.best_dev_epoch
        f.close()

    def restore_parameters(self):
        f = shelve.open(self.model_save_path + "model_obj.shv")
        self.configure = f['configure']
        self.best_dev_f1 = f['best_dev_f1']
        self.best_dev_epoch = f['best_dev_epoch']
        f.close()

    def load_previous_model(self):
        self.restore_parameters()
        if self.best_dev_epoch != 0:
            self.load_data_and_embedding()
            self.create_model()
            self.restore_tf_model(self.best_dev_epoch)
            print "load trained model at epoch: ", self.epoch, self.best_dev_epoch
            print "current model dev F1: ", self.best_dev_f1
        else:
            print "No previous model available."
            exit(-1)

    def create_model(self):

        char_embeddings = tf.Variable(self.char_embeddings.get_embeddings(), dtype=tf.float32, name="char_embeddings")
        word_embeddings = tf.Variable(self.word_embeddings.get_embeddings(), dtype=tf.float32, name="word_embeddings")
        char_pos_embeddings = tf.Variable(
            tf.random_uniform([self.max_char_len * 2, self.position_embedding_dim], -0.1, 0.1), dtype=tf.float32,
            name="char_position_embeddings")
        word_pos_embeddings = tf.Variable(
            tf.random_uniform([self.max_word_len * 2, self.position_embedding_dim], -0.1, 0.1), dtype=tf.float32,
            name="word_position_embeddings")

        char_ids = tf.placeholder(tf.int32, [None, self.max_char_len])
        word_ids = tf.placeholder(tf.int32, [None, self.max_word_len])
        char_seq_len = tf.placeholder(tf.int32, [None])
        word_seq_len = tf.placeholder(tf.int32, [None])

        char_lex_ctx_ids = tf.placeholder(tf.int32, [None, self.char_win_size * 2 + 1])
        word_lex_ctx_ids = tf.placeholder(tf.int32, [None, self.word_win_size * 2 + 1])

        char_position_ids = tf.placeholder(tf.int32, [None, self.max_char_len])
        word_position_ids = tf.placeholder(tf.int32, [None, self.max_word_len])

        positive_loss_indicator = tf.placeholder(tf.float32, [None])

        reg_label_ids = tf.placeholder(tf.int32, [None])
        cls_label_ids = tf.placeholder(tf.int32, [None])
        is_train = tf.placeholder(tf.bool, [])

        embed_char = self.embed(char_embeddings, char_ids)  # B*T*char_embedding_dim
        embed_word = self.embed(word_embeddings, word_ids)  # B*T*word_embedding_dim
        embed_char_pos = self.embed(char_pos_embeddings, char_position_ids)  # B*T*position_embedding_dim
        embed_word_pos = self.embed(word_pos_embeddings, word_position_ids)  # B*T*position_embedding_dim
        embed_char_lex = self.embed(char_embeddings, char_lex_ctx_ids)  # B*(2*char_win_size+1)*char_embed_dim
        embed_word_lex = self.embed(word_embeddings, word_lex_ctx_ids)  # B*(2*word_win_size+1)*word_embed_dim

        char_lexical_features = tf.reshape(embed_char_lex,
                                           shape=[-1, (2 * self.char_win_size + 1) * self.char_embedding_dim])
        word_lexical_features = tf.reshape(embed_word_lex,
                                           shape=[-1, (2 * self.word_win_size + 1) * self.word_embedding_dim])

        concat_char_embedding = tf.concat([embed_char, embed_char_pos],
                                          axis=2)  # B*T*(char_embedding_dim+position_embedding_dim)
        concat_word_embedding = tf.concat([embed_word, embed_word_pos],
                                          axis=2)  # B*T*(word_embedding_dim+position_embedding_dim)

        char_input_mask_layer = MaskLayer("char_input_mask")
        char_input_mask_layer.set_extra_parameters({"mask_value": 0})
        masked_concat_char_embedding = char_input_mask_layer(concat_char_embedding, char_seq_len)

        word_input_mask_layer = MaskLayer("word_input_mask")
        word_input_mask_layer.set_extra_parameters({"mask_value": 0})
        masked_concat_word_embedding = word_input_mask_layer(concat_word_embedding, word_seq_len)

        char_conv_layer = Conv1DLayer("char_conv1d", self.char_feature_map_size)
        char_feature_maps = tf.tanh(char_conv_layer(masked_concat_char_embedding))  # B*T*char_feature_map_size

        word_conv_layer = Conv1DLayer("word_conv1d", self.word_feature_map_size)
        word_feature_maps = tf.tanh(word_conv_layer(masked_concat_word_embedding))  # B*T*word_feature_map_size

        max_pooled_char_maps = tf.reduce_max(char_feature_maps, axis=1)  # B * char_feature_map
        max_pooled_word_maps = tf.reduce_max(word_feature_maps, axis=1)  # B * word_feature_map

        latent_char_dense_layer = DenseLayer("char_dense_layer",self.gate_vec_size)
        latent_word_dense_layer = DenseLayer("word_dense_layer",self.gate_vec_size)

        char_features = latent_char_dense_layer(tf.concat([max_pooled_char_maps, char_lexical_features], axis=1) )
        word_features = latent_word_dense_layer(tf.concat([max_pooled_word_maps, word_lexical_features], axis=1) )

        reg_gate_linear = DenseLayer("reg_gate_linear",self.gate_vec_size)
        reg_gate_value = tf.sigmoid(reg_gate_linear(tf.concat([char_features,word_features],axis = 1)))
        reg_features = char_features * reg_gate_value + word_features * (1-reg_gate_value)
        
        cls_features = char_features * reg_gate_value + word_features * (1-reg_gate_value)

        # dropout_features = features
        dropout_reg_features = tf.layers.dropout(reg_features, training=is_train)
        dropout_cls_features = tf.layers.dropout(cls_features, training=is_train)

        reg_output_layer = DenseLayer("reg_output_layer", self.reg_label_size)
        cls_output_layer = DenseLayer("cls_output_layer", self.cls_label_size)

        reg_logit = reg_output_layer(dropout_reg_features)
        cls_logit = cls_output_layer(dropout_cls_features)

        reg_loss_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=reg_label_ids, logits=reg_logit)
        cls_loss_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=cls_label_ids,
                                                                         logits=cls_logit) * positive_loss_indicator
        reg_loss = tf.reduce_mean(reg_loss_sample)
        cls_loss = tf.reduce_sum(cls_loss_sample) / (1e-8 + tf.reduce_sum(positive_loss_indicator) )
        total_loss = reg_loss + cls_loss


        #basic_optimizer = tf.train.AdamOptimizer(learning_rate=1, epsilon=1e-06)
        basic_optimizer = tf.train.AdadeltaOptimizer(learning_rate=1, epsilon=1e-06)
        clipped_optimizer = basic_optimizer

        '''
        clip_dict = {}
        for var in tf.trainable_variables():
            if "weight" in var.name:
                clip_dict[var] = [0]

        clipped_optimizer = tf.contrib.opt.VariableClippingOptimizer(basic_optimizer,clip_dict,9)
        '''

        # train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
        train_step = clipped_optimizer.minimize(total_loss)
        # train_step = tf.train.GradientDescentOptimizer(0.1).minimize(total_loss)

        with tf.control_dependencies([train_step]):
            clip_ops = []
            for var in tf.trainable_variables():
                if "weight" in var.name:
                    clip_ops.append( tf.assign(var,tf.clip_by_norm(var,9,axes = [0])) )
            train_step_with_clip = tf.group(*clip_ops)

        self.variables['reg_logit'] = reg_logit
        self.variables['cls_logit'] = cls_logit
        self.variables['reg_loss'] = reg_loss
        self.variables['cls_loss'] = cls_loss
        self.variables['total_loss'] = total_loss
        self.variables['train_step'] = train_step_with_clip
        self.variables['reg_logit'] = reg_logit
        #self.variables['clip_op'] = clip_op

        self.placeholders['char_sents'] = char_ids
        self.placeholders['char_seq_len'] = char_seq_len
        self.placeholders['char_ctx'] = char_lex_ctx_ids
        self.placeholders['word_sents'] = word_ids
        self.placeholders['word_seq_len'] = word_seq_len
        self.placeholders['word_ctx'] = word_lex_ctx_ids
        self.placeholders['char_position'] = char_position_ids
        self.placeholders['word_position'] = word_position_ids
        self.placeholders['is_positive'] = positive_loss_indicator
        self.placeholders['reg_label_ids'] = reg_label_ids
        self.placeholders['cls_label_ids'] = cls_label_ids
        self.placeholders['is_train'] = is_train

        print
        print "Model Variables: "
        for var in tf.trainable_variables():
            print var.name,var.get_shape().as_list()
        print "---------------------------------"
        

    def save_tf_model(self):
        saver = tf.train.Saver()
        saver.save(self.sess, "model/model.ckpt", global_step=self.epoch)

    def restore_tf_model(self, epoch):
        self.epoch = epoch
        saver = tf.train.Saver()
        saver.restore(self.sess, "model/model.ckpt-" + str(epoch))

    def train_model(self):
        init = tf.global_variables_initializer()
        sess = self.sess
        if self.epoch == 0:
            sess.run(init)
        while (1):
            self.epoch += 1
            epoch = self.epoch
            epoch_loss = 0.0
            epoch_reg_loss = 0.0
            epoch_cls_loss = 0.0
            for batch in self.train_data.next_train_batch(self.batch_size):
                batch['is_train'] = True
                feed_dict = {}
                for key in self.placeholders:
                    feed_dict[self.placeholders[key]] = batch[key]
                if self.epoch == 1:
                    opt_vars = [self.variables[var_name] for var_name in
                                ['total_loss', 'reg_loss', 'cls_loss', 'cls_loss']]
                else:
                    opt_vars = [self.variables[var_name] for var_name in
                                ['total_loss', 'reg_loss', 'cls_loss', 'train_step']]
                loss, reg_loss, cls_loss, _ = sess.run(opt_vars, feed_dict=feed_dict)
                #print "start clip"
                #_ = sess.run(self.variables['clip_op'])
                #print "end clip"
                # print loss
                epoch_loss += loss * batch['cnt']
                epoch_reg_loss += reg_loss * batch['cnt']
                epoch_cls_loss += cls_loss * batch['cnt']

            print "epoch ", epoch, "loss: ", epoch_loss, epoch_reg_loss, epoch_cls_loss
            if epoch !=1:
                print
                print "Dev data performance at epoch %d" % (epoch)
                results = self.core_evaluate(self.decode(self.dev_data)[0], self.dev_data)
                s_p,s_r,s_f = results[0]
                t_p,t_r,t_f = results[1]
                print "Span result: ",s_p,s_r,s_f
                print "Type Result: ",t_p,t_r,t_f
                print
                print "Test data performance at epoch %d" %(epoch)
                results = self.core_evaluate(self.decode(self.test_data)[0], self.test_data)
                s_p,s_r,s_f = results[0]
                t_p,t_r,t_f = results[1]
                print "Span result: ",s_p,s_r,s_f
                print "Type Result: ",t_p,t_r,t_f
                print "---------------------"
            sys.stdout.flush()
            
    def decode(self, data, readout=True):
        rst = set()
        rst_to_prob = {}
        for batch in data.next_test_batch(1000):
            batch['is_train'] = False
            feed_dict = {}
            for key in self.placeholders:
                feed_dict[self.placeholders[key]] = batch[key]
            opt_vars = [self.variables[var_name] for var_name in ['reg_logit', 'cls_logit']]
            reg_logit, cls_logit = self.sess.run(opt_vars, feed_dict=feed_dict)

            reg_labels = np.argmax(reg_logit, axis=1).tolist()
            cls_labels = np.argmax(cls_logit, axis=1).tolist()
            exp_reg_logit = np.exp(reg_logit)
            reg_probs = exp_reg_logit / np.sum(exp_reg_logit, axis=1, keepdims=True)
            reg_probs = [reg_probs[i] for i in enumerate(reg_labels)]
            exp_cls_logit = np.exp(cls_logit)
            cls_probs = exp_cls_logit / np.sum(exp_cls_logit, axis=1, keepdims=True)
            cls_probs = [cls_probs[i] for i in enumerate(cls_labels)]

            assert len(reg_labels) == len(cls_labels)
            assert len(reg_labels) == batch['cnt']

            for data_key, reg_label, cls_label, reg_prob, cls_prob in zip(batch['keys'], reg_labels, cls_labels,
                                                                          reg_probs, cls_probs):
                doc_id, sent_id, anchor, _, _ = data_key
                if reg_label != 6:
                    offset, length = data.anchor_reg_to_offset_length(doc_id, sent_id, anchor, reg_label)
                    rst_tuple = (doc_id, offset, length, data.id2label[cls_label])
                    rst.add(rst_tuple)
                    if rst_tuple not in rst_to_prob or rst_to_prob[rst_tuple][0] < reg_prob:
                        rst_to_prob[rst_tuple] = (reg_prob, cls_prob)
        if readout:
            output = open(self.result_output_path + data.split + "_" + str(self.epoch) + ".rst", "w")
            sorted_rst = sorted(rst, key=lambda x: (x[0], x[1]), reverse=False)
            for tup in sorted_rst:
                tup_l = [str(i) for i in tup]
                tup_l.append(str(rst_to_prob[tup][0]))
                tup_l.append(str(rst_to_prob[tup][1]))
                output.write("\t".join(tup_l) + "\n")
            output.close()
        return rst, rst_to_prob

    def evaluate(self, rst, data):
        tp_span = 0.0
        tp_typed = 0.0
        if len(rst) <1:
            return(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        for doc_id, offset, length, label in rst:
            key = (doc_id, offset, length)
            if key in data.golden:
                tp_span += 1
                if label in data.golden[key]:
                    tp_typed += 1
        if tp_span <1 or tp_typed <1:
            return (0.0,0.0,0.0),(0.0,0.0,0.0)
        p_span = tp_span / len(rst)
        r_span = tp_span / len(data.golden)
        f_span = p_span * r_span * 2 / (p_span + r_span)

        p_typed = tp_typed / len(rst)
        r_typed = tp_typed / len(data.golden)
        f_typed = p_typed * r_typed * 2 / (p_typed + r_typed)

        return (p_span, r_span, f_span), (p_typed, r_typed, f_typed)

    def core_evaluate(self,rst,data):
        key2label = {}
        for doc_id,offset,length,label in rst:
            key2label[(doc_id,offset,length)] = [label]

        output_strs = transform_to_score_list(key2label)
        eval_rst = score(data.golden_strs,output_strs)[2]
        r1 = eval_rst['plain']['micro']
        r2 = eval_rst['mention_type']['micro']
        #r3 = score(data.golden_strs,output_strs)[2]['realis_status']['micro']
        return r1,r2


    def embed(self, params, ids):
        return tf.nn.embedding_lookup(params, ids)

    def output_result(self, key2label, epoch, prefix):
        output = open("tmp/" + prefix + "_result." + str(epoch) + ".rst", "w")
        for key in key2label:
            output.write(
                str(key[0]) + "\t" + str(key[1]) + "\t" + str(key[2]) + "\t" + str(key[3]) + "\t" + str(key[4]) + "\t ")
            for label, prob in key2label[key]:
                output.write(label + "\t" + str(prob) + "\t")
            output.write("\n")
        output.close()

    def load_data_and_embedding(self):

        self.train_data = EEData("train", self.configure)
        self.test_data = EEData("test", self.configure)
        self.dev_data = EEData("dev", self.configure)

        word2cnt = self.train_data.get_word_to_cnt()
        char2cnt = self.train_data.get_char_to_cnt()

        self.word2cnt = self.train_data.get_word_to_cnt()
        self.char2cnt = self.train_data.get_char_to_cnt()

        self.load_init_embeddings()

        self.train_data.translate_sentence(self.char_embeddings, self.word_embeddings)
        self.dev_data.translate_sentence(self.char_embeddings, self.word_embeddings)
        self.test_data.translate_sentence(self.char_embeddings, self.word_embeddings)

    def load_init_embeddings(self):
        self.char_embeddings = GloveEmbeddings(self.char_embeddings_file, self.char2cnt)
        self.word_embeddings = GloveEmbeddings(self.word_embeddings_file, self.word2cnt)


if __name__ == "__main__":
    confs = Configure(sys.argv[1])
    # model_save_path = "model/"
    # dev_ret_path = "tmp/"
    model = NPN(conf=confs)
    model.load_data_and_embedding()
    model.create_model()
    model.train_model()
    # model.save_parameters()
    # model.train_model()
