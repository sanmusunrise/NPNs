from tfnlp.embedding.GloveEmbeddings import *
import random
from scorer.event_scorer import *
from Configure import *

class EEData:

    def __init__(self,split,confs =None):
        self.train_positive= {} #(doc_id,sent_id) ->[([(anchor_id,span_type)],label_id)]
        self.trim_train_positive = []   #(doc_id,sent_id,anchor_id,span_type,label_id)
        self.trim_train_negative = []
        self.train_overlapped_negative = {} #(doc_id,sent_id,anchor) ->[span_type]
        self.train_negative = {}    #(doc_id,sent_id)->[(anchor,span_type)]
        self.char_sents_str = {}
        self.char_sents_id = {}
        self.word_sents_str = {}
        self.word_sents_id = {}

        self.golden = {}
        self.golden_strs = None

        self.split = split
        self.id2label = {}
        self.label2id = {}
        self.anchor2id = {(0,1):0,(0,2):1,(1,2):2,(0,3):3,(1,3):4,(2,3):5}
        self.reg_to_bias_length ={0:(0,1),1:(0,2),2:(-1,2),3:(0,3),4:(-1,3),5:(-2,3)}

        self.max_char_len = confs['max_char_len'] if confs else 200
        self.max_word_len = confs['max_word_len'] if confs else 100
        self.char_win_size = confs['char_win_size'] if confs else 3
        self.word_win_size = confs['word_win_size'] if confs else 1
        self.data_dir = confs['data_dir'] if confs else "../trigger_data"
        self.char_embeddings = None
        self.word_embeddings = None
        self.confs = confs

        if split == "train":
            self.is_train = True
        else:
            self.is_train = False

        self.load(split)
        self.golden_strs = transform_to_score_list(self.golden)

        #print score(self.golden_strs,self.golden_strs[:200])[2]

    def get_char_offset(self,doc_id,sent_id,char_anchor):
        return self.char_sents_str[(doc_id,sent_id)][char_anchor][2]
    def get_char_word(self,doc_id,sent_id,char_anchor):
        return self.char_sents_str[(doc_id, sent_id)][char_anchor][1]

    def anchor_reg_to_offset_length(self,doc_id,sent_id,anchor,reg_type):
        anchor_offset = self.get_char_offset(doc_id,sent_id,anchor)
        bias,length = self.reg_to_bias_length[reg_type]
        offset = anchor_offset + bias
        return offset,length

    def trim_data(self):
        self.trim_train_negative = []
        self.trim_train_positive = []
        triple_dict = set()
        for sent_key in self.train_positive:
            for spans,label_id in self.train_positive[sent_key]:
                for anchor,span_type in spans:
                    self.trim_train_positive.append([sent_key[0],sent_key[1],anchor,span_type,label_id])
                    triple_dict.add((sent_key,anchor))

        for sent_key in self.train_negative:
            for anchor,span_type in self.train_negative[sent_key]:
                if (sent_key,anchor) in triple_dict:
                    #print "Already found: ",sent_key,anchor
                    continue
                self.trim_train_negative.append([sent_key[0],sent_key[1],anchor,6,None])
                triple_dict.add((sent_key,anchor))

        return len(self.trim_train_positive),len(self.trim_train_negative)



    def next_train_batch(self,batch_size = 128):
        ratio = self.confs['train_negative_ratio']
        random.shuffle(self.trim_train_positive)
        random.shuffle(self.trim_train_negative)
        neg_idx = 0
        batch = self.empty_batch()
        for postive_data in self.trim_train_positive:
            cnt = self.add_data_to_batch(batch,postive_data)
            if cnt >= batch_size:
                yield batch
                batch = self.empty_batch()
                continue
            for i in xrange(ratio):

                cnt = self.add_data_to_batch(batch,self.trim_train_negative[neg_idx])
                neg_idx += 1
                if neg_idx >= len(self.trim_train_negative):
                    neg_idx = 0
                    random.shuffle(self.trim_train_negative)
                if cnt >= batch_size:
                    yield batch
                    batch = self.empty_batch()
                    continue
        if batch['cnt'] !=0:
            yield batch

    def next_test_batch(self,batch_size = 128):
        batch = self.empty_batch()
        for doc_id,sent_id in self.char_sents_str:
            for anchor in xrange(len(self.char_sents_str[(doc_id,sent_id)])):
                data_tuple = (doc_id,sent_id,anchor,6,9)
                cnt = self.add_data_to_batch(batch,data_tuple)
                if cnt >= batch_size:
                    yield batch
                    batch = self.empty_batch()
        if batch['cnt'] !=0:
            yield batch

    def add_data_to_batch(self,batch,data_tuple):
        doc_id, sent_id, anchor, span_type, label_id = data_tuple
        batch['cnt'] +=1
        batch['char_sents'].append(self.char_sents_id[(doc_id,sent_id)][0])
        batch['char_seq_len'].append(self.char_sents_id[(doc_id,sent_id)][1])

        char_ctx = []
        for ctx_idx in xrange(anchor - self.char_win_size,anchor + self.char_win_size +1):
            if ctx_idx <0 or ctx_idx >= self.max_char_len:
                char_ctx.append(self.char_embeddings.get_padding_id())
            else:
                char_ctx.append(self.char_sents_id[(doc_id,sent_id)][0][ctx_idx])
        batch['char_ctx'].append(char_ctx)

        batch['word_sents'].append(self.word_sents_id[(doc_id,sent_id)][0])
        batch['word_seq_len'].append(self.word_sents_id[(doc_id,sent_id)][1])

        corres_word_id = self.get_char_word(doc_id, sent_id, anchor)
        word_ctx = []
        for ctx_idx in xrange(corres_word_id - self.word_win_size,corres_word_id + self.word_win_size +1):
            if ctx_idx <0 or ctx_idx >= self.max_word_len:
                word_ctx.append(self.word_embeddings.get_padding_id())
            else:
                word_ctx.append(self.word_sents_id[(doc_id,sent_id)][0][ctx_idx])
        batch['word_ctx'].append(word_ctx)

        batch['char_position'].append(self.get_relative_position(anchor,self.max_char_len))
        batch['word_position'].append(self.get_relative_position(corres_word_id,self.max_word_len))
        if label_id ==None:
            batch['is_positive'].append(0.0)
        else:
            batch['is_positive'].append(1.0)
        batch['reg_label_ids'].append(span_type)
        if label_id ==None:
            batch['cls_label_ids'].append(9)
        else:
            batch['cls_label_ids'].append(label_id)
        batch['keys'].append(data_tuple)

        return batch['cnt']




    def empty_batch(self):
        batch = {}
        batch['cnt'] = 0
        batch['char_sents'] = []
        batch['char_seq_len'] = []
        batch['char_ctx'] = []
        batch['word_sents'] = []
        batch['word_seq_len'] = []
        batch['word_ctx'] = []
        batch['char_position'] = []
        batch['word_position'] = []
        batch['is_positive'] = []
        batch['reg_label_ids'] = []
        batch['cls_label_ids'] = []
        batch['keys'] = []  #batch *(doc_id,sent_id,anchor)

        return batch

    def check_length(self):
        for sent_key in self.char_sents_str:
            char_len = len(self.char_sents_str[sent_key])
            word_len = len(self.word_sents_str[sent_key])

            if char_len > self.max_char_len:
                print sent_key,char_len,word_len,1
            elif word_len > self.max_word_len:
                print sent_key,char_len,word_len,2

    def clip_sentence(self):
        '''clip sentence exceeded the max_length
        '''

        for sent_key in self.char_sents_str:
            char_sent = self.char_sents_str[sent_key]
            word_sent = self.word_sents_str[sent_key]

            char_len = len(char_sent)
            word_len = len(word_sent)


            if char_len < self.max_char_len and word_len < self.max_word_len:
                continue
            if word_len > self.max_word_len:
                #print "Clip word sent",sent_key,word_len,self.is_train
                word_sent = word_sent[:self.max_word_len]
                char_boundary = None
                for idx,(char_token,word_id,offset) in enumerate(char_sent):
                    if word_id >= self.max_word_len:
                        char_boundary = idx
                        break
                assert char_boundary != None
                #print char_boundary,char_len
                char_sent = char_sent[:char_boundary]

            char_sent = char_sent[:self.max_char_len]
            self.char_sents_str[sent_key] = char_sent
            self.word_sents_str[sent_key] = word_sent


    def load(self,data_split = "train"):
        data_dir = self.data_dir

        ids_file = data_dir + data_split + "/"  + data_split + ".ids.dat"
        char_file  = data_dir + data_split + "/" + data_split + ".char.dat"
        word_file = data_dir + data_split + "/" + data_split + ".word.dat"
        golden_file = data_dir + data_split + "/" +  data_split + ".golden.dat"
        label2id_file = data_dir + "label2id.dat"
        self.load_label2id(label2id_file)

        self.load_chars(char_file)
        self.load_words(word_file)
        self.load_golden(golden_file)
        self.clip_sentence()

        if data_split =="train":
            self.load_train_positive(ids_file)
            self.generate_train_negative()
            self.trim_data()

    def load_label2id(self,file_name):
        self.id2label = {}
        self.label2id = {}
        for line in open(file_name):
            label,i = line.strip().split()
            self.id2label[int(i)] = label
            self.label2id[label] = int(i)

        return len(self.id2label),len(self.label2id)

    def load_train_positive(self,ids_file):
        for line in open(ids_file):
            line = line.decode("utf-8").strip().split("\t")
            if len(line) !=8:
                #print line
                #print len(line)
                continue
            doc_id,sent_id,begin_char_id,offset,length,train_label,_,_ = line
            sent_id = int(sent_id)
            begin_char_id = int(begin_char_id)
            offset = int(offset)
            length = int(length)
            if length >3:
                #print "A length > 3 trigger!"
                continue

            if not (doc_id,sent_id) in self.train_positive:
                self.train_positive[(doc_id,sent_id)] = []
            spans = []
            for bias in xrange(length):
                anchor_char = begin_char_id + bias
                if anchor_char > len(self.char_sents_str[(doc_id,sent_id)]):
                    #print "Warning: a trigger is clipped in ",doc_id,sent_id,anchor_char,len(self.char_sents_str[(doc_id,sent_id)])
                    continue
                #print self.char_sents_str[(doc_id,sent_id)][anchor_char][0].encode("utf-8"),
                span_type = self.anchor2id[(bias,length)]
                spans.append((anchor_char,span_type))
            #print doc_id,sent_id,offset,length,train_label
            self.train_positive[(doc_id, sent_id)].append((spans,self.label2id[train_label]))

    def load_chars(self,char_file):
        for line in open(char_file):
            line = line.decode("utf-8").strip().split("\t")
            doc_id,sent_id,contains_trigger,tokens = line
            sent_id = int(sent_id)
            #if contains_trigger == "False":
            #    continue
            tokens = tokens.split(" ")
            sent = []
            for token in tokens:
                token = token.split("|||")
                sent.append((token[0],int(token[2]),int(token[1]))) #char_token,word_id,offset
            self.char_sents_str[(doc_id,sent_id)] = sent

    def load_words(self,word_file):
        for line in open(word_file):
            line = line.decode("utf-8").strip().split("\t")
            doc_id,sent_id,contains_trigger,tokens = line
            sent_id = int(sent_id)
            #if contains_trigger == "False":
            #    continue
            tokens = tokens.split(" ")
            sent = []
            for token in tokens:
                token = token.split("|||")
                sent.append((token[0],int(token[1]),int(token[2]),int(token[3]))) #word_token,word_begin,word_end,offset
            self.word_sents_str[(doc_id,sent_id)] = sent

    def load_golden(self,golden_file):
        for line in open(golden_file):
            line = line.strip().split("\t")
            key = line[0],int(line[1]),int(line[2])
            if not key in self.golden:
                self.golden[key] = []
            self.golden[key].append(line[4])

    def generate_train_negative(self):
        positive_map = {}
        for doc_id,sent_id in self.train_positive:
            for spans,_ in self.train_positive[(doc_id,sent_id)]:
                for anchor,span_type in spans:
                    positive_map[(doc_id,sent_id,anchor)] = span_type

        for doc_id,sent_id in self.char_sents_str:
            for anchor in xrange(len(self.char_sents_str[(doc_id,sent_id)])):
                if (doc_id,sent_id,anchor) in positive_map:
                    self.train_overlapped_negative[(doc_id,sent_id,anchor)] = []
                    for span_type in xrange(len(self.anchor2id)):
                        if span_type == positive_map[(doc_id,sent_id,anchor)]:
                            continue
                        self.train_overlapped_negative[(doc_id,sent_id,anchor)].append(span_type)
                else:
                    if not (doc_id,sent_id) in self.train_negative:
                        self.train_negative[(doc_id,sent_id)] = []
                    for span_type in xrange(len(self.anchor2id)):
                        self.train_negative[(doc_id,sent_id)].append((anchor,span_type))

    def get_relative_position(self,token_id,max_len):
        anchor = [i+max_len- token_id -1 for i in xrange(0,max_len)]
        return anchor

    def positive_size(self):
        total = 0
        total_spans = 0
        # (doc_id,sent_id) ->[([(anchor_id,span_type)],label_id)]
        for doc_id,sent_id in self.train_positive:
            total += len(self.train_positive[(doc_id,sent_id)])
            #print doc_id,sent_id, self.train_positive[(doc_id,sent_id)]
            for spans,label in self.train_positive[(doc_id,sent_id)]:
                total_spans += len(spans)
        return total,total_spans

    def negative_overlap_size(self):
        total = 0.0
        for doc_id,sent_id,anchor in self.train_overlapped_negative:
            total += len(self.train_overlapped_negative[(doc_id,sent_id,anchor)])
        return len(self.train_overlapped_negative),total

    def negative_size(self):
        total = 0
        for (doc_id,sent_id),instances in self.train_negative.iteritems():
            total += len(instances)
        return total

    def golden_size(self):
        total =0
        for key in self.golden:
            total += len(self.golden[key])
        return total

    def sent_size(self):
        c_total = len(self.char_sents_str)
        w_total = len(self.word_sents_str)
        assert c_total ==w_total
        return c_total,w_total

    def size(self):
        ret = []
        ret.append(("positive_size",self.positive_size()))
        ret.append(("negative_overlap_size",self.negative_overlap_size()))
        ret.append(("negative_size",self.negative_size()))
        ret.append(("golden_size",self.golden_size()))
        ret.append(("sent_size",self.sent_size()))
        return ret

    def translate_sentence(self,char_embeddings,word_embeddings,padding = True):
        self.char_embeddings = char_embeddings
        self.word_embeddings = word_embeddings

        self.word_sents_id = {}
        self.char_sents_id = {}
        for sent_key in self.char_sents_str:
            sent = [i[0] for i in self.char_sents_str[sent_key]]
            char_ids,length = char_embeddings.words_to_ids(sent,self.max_char_len,padding)
            #print char_ids
            #print
            self.char_sents_id[sent_key] = (char_ids,length)
            #print "char: ", sent_key, len(sent), len(char_ids), length

        for sent_key in self.word_sents_str:
            sent = [i[0] for i in self.word_sents_str[sent_key] ]
            word_ids,length = word_embeddings.words_to_ids(sent,self.max_word_len,padding)
            self.word_sents_id[sent_key] = (word_ids,length)
            #print "word: ",sent_key,len(sent),len(word_ids),length

        return (len(self.word_sents_id),len(self.word_sents_str)),(len(self.char_sents_id),len(self.char_sents_str))


    def get_char_to_cnt(self,char2cnt = None):
        ''' Get char-level word count from this batch

        return: {char->cnt}
        '''
        if not char2cnt:
            char2cnt = {}
        for sent_key in self.char_sents_str:
            for token in self.char_sents_str[sent_key]:
                ch = token[0]
                if ch not in char2cnt:
                    char2cnt[ch] = 0
                char2cnt[ch] +=1
        return char2cnt


    def get_word_to_cnt(self,word2cnt = None):
        ''' Get word-level word count from this batch

        return: {word->cnt}
        '''
        if not word2cnt:
            word2cnt = {}
        for sent_key in self.word_sents_str:
            for token in self.word_sents_str[sent_key]:
                ch = token[0]
                if ch not in word2cnt:
                    word2cnt[ch] =0
                word2cnt[ch] +=1

        return word2cnt

if __name__ =="__main__":
    confs = Configure("config.cfg")
    train_data = EEData("train",confs)
    test_data = EEData("test",confs)
    dev_data = EEData("dev",confs)


    print train_data.size()
    print test_data.size()
    print dev_data.size()


    word2cnt = train_data.get_word_to_cnt()
    char2cnt = train_data.get_char_to_cnt()

    word_embed = GloveEmbeddings("word_word2vec.dat", word2cnt)
    char_embed = GloveEmbeddings("char_word2vec.dat", char2cnt)


    print train_data.translate_sentence(char_embeddings=char_embed,word_embeddings=word_embed)
    print test_data.translate_sentence(char_embeddings=char_embed,word_embeddings=word_embed)
    print dev_data.translate_sentence(char_embeddings=char_embed,word_embeddings=word_embed)

    print train_data.trim_data()
    print train_data.get_relative_position(0,train_data.max_word_len)
    print train_data.get_relative_position( train_data.max_word_len-1, train_data.max_word_len)



    for batch in test_data.next_test_batch(1):
        print batch['cnt']
        batch['cnt'] = 0

        print "word_position", batch['word_position']
        '''
        #print "char_sent", batch['char_sents']
        print "char_seq_len", batch['char_seq_len']
        #print "word_sent",batch['word_sents']
        print "word_seq_len",batch['word_seq_len']
        print "char_position",batch['char_position']
        print "word_position",batch['word_position']
        print "is_positive",batch['is_positive']
        print "reg_label_ids",batch['reg_label_ids']
        print "cls_label_ids",batch['cls_label_ids']
        print "key",batch['keys']
        '''
