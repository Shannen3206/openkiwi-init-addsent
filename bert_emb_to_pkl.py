import pickle
import numpy as np

# path = '/home/snchen/project/openkiwi-init-addsent/data_polyglot/embeddings2/de/words_embeddings_32.pkl'
# f = open(path, 'rb')
# data = pickle.load(f, encoding='latin1')
# print(data)

"""源端"""
path_en_emb = '/home/snchen/project/bert-as-service/data_embeddings/exp2/en_vocab.npy'
source_bert = []
src_bert = np.load(path_en_emb, allow_pickle=True).item()
for s in src_bert:
    source_bert.append(list(src_bert[s].squeeze(0)))
source_bert = np.array(source_bert)

path_en_data = '/home/snchen/project/bert-as-service/data_embeddings/exp2/en_vocab.txt'
source_data = []
src_data = open(path_en_data, 'r', encoding='utf-8')
for s in src_data:
    source_data.append(s)
source_data = tuple(source_data)

path_en = '/home/snchen/project/bert-as-service/data_embeddings/exp2/en_word_embed.pkl'
data_en = (source_data, source_bert)
f_en = open(path_en, 'wb')
pickle.dump(data_en, f_en)

"""目标端"""
path_de_emb = '/home/snchen/project/bert-as-service/data_embeddings/exp2/de_vocab.npy'
target_bert = []
tgt_bert = np.load(path_de_emb, allow_pickle=True).item()
for t in tgt_bert:
    target_bert.append(list(tgt_bert[t].squeeze(0)))
target_bert = np.array(target_bert)

path_de_data = '/home/snchen/project/bert-as-service/data_embeddings/exp2/de_vocab.txt'
target_data = []
tgt_data = open(path_de_data, 'r', encoding='utf-8')
for t in tgt_data:
    target_data.append(t)
target_data = tuple(target_data)

path_de = '/home/snchen/project/bert-as-service/data_embeddings/exp2/de_word_embed.pkl'
data_de = (target_data, target_bert)
f_de = open(path_de, 'wb')
pickle.dump(data_de, f_de)
