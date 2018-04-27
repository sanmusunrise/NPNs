from WordEmbeddings import *
import sys

class GloveEmbeddings(WordEmbeddings):
    def load_from_file(self,file_name):
        for line in open(file_name):
            line = line.strip().split()
            vec = [float(i) for i in line[1:]]
            #print line[0]
            try:
                word = line[0].decode("utf-8")
            except:
                print "unable to decode using utf-8"
                continue

            self.id2word[len(self.id2word)] = word
            self.word2id[word] = len(self.word2id)
            self.embeddings.append(vec)
        return len(self.embeddings),len(self.embeddings[1])
    
    def load_based_word2cnt(self,file_name,word2cnt,max_vocab_size = None):
        word2vec = {}
        for line in open(file_name):
            line = line.strip().split()
            try: 
                word = line[0].decode("utf-8")
            except:
                print "unable to decode using utf-8"
                continue
            
            if word in word2cnt:
                word2vec[word] = [float(i) for i in line[1:]]
        dim = len(word2vec.items()[0][1])
        sorted_word2cnt = sorted(word2cnt.items(),key = lambda x:x[1],reverse = True)
        for word,cnt in sorted_word2cnt:
            if word in word2vec:
                self.embeddings.append(word2vec[word])
                #print word
            else:
                self.embeddings.append(self.get_init_vector(dim))
            self.id2word[len(self.id2word)] = word
            self.word2id[word] = len(self.word2id)

        return len(self.embeddings),len(self.embeddings[1])


if __name__ =="__main__":
    #emd = GloveEmbeddings(file_name = "../../data_ori/glove.840B.300d.txt")
    embed = GloveEmbeddings(sys.argv[1])
    for i,word in embed.id2word.items():
        print i,word.encode("utf-8")
