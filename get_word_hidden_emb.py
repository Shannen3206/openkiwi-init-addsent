import torch
import pickle
import numpy as np
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from transformers import RobertaTokenizer, RobertaModel

en_path = '/home/snchen/project/openkiwi-init-addsent/data_embeddings/exp3/en.txt'
de_path = '/home/snchen/project/openkiwi-init-addsent/data_embeddings/exp2/de.txt'
save_en = '/home/snchen/project/openkiwi-init-addsent/data_embeddings/exp3/en_word_embed.pkl'
save_de = '/home/snchen/project/openkiwi-init-addsent/data_embeddings/exp3/de_word_embed.pkl'

en_data, de_data = [], []
en_bert, de_bert = [], []

"""start:源端"""
with open(en_path, 'r', encoding='utf-8') as f_en:
    for item in f_en:
        item = item.split('\n')[0]
        en_data.append(item)
en_data = tuple(en_data)

# tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
# model = BertModel.from_pretrained('bert-large-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')

for i, item in enumerate(en_data):
    print(item)
    tokenized_text = tokenizer.tokenize(item)                         # token初始化
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)            # 获取词汇表索引
    tokens_tensor = torch.tensor([indexed_tokens])  # 将输入转化为torch的tensor
    with torch.no_grad():                           # 禁用梯度计算 因为只是前向传播获取隐藏层状态，所以不需要计算梯度
        last_hidden_states = (model(tokens_tensor)[1].squeeze(0)).tolist()
        en_bert.append(list(last_hidden_states))
en_bert = np.array(en_bert)

source_data = (en_data, en_bert)
with open(save_en, 'wb') as file_en:
    pickle.dump(source_data, file_en)
"""end"""

"""start:目标端"""
with open(de_path, 'r', encoding='utf-8') as f_de:
    for item in f_de:
        item = item.split('\n')[0]
        de_data.append(item)
de_data = tuple(de_data)

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')

for i, item in enumerate(de_data):
    print(item)
    tokenized_text = tokenizer.tokenize(item)                         # token初始化
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)  # 获取词汇表索引
    tokens_tensor = torch.tensor([indexed_tokens])  # 将输入转化为torch的tensor
    with torch.no_grad():  # 禁用梯度计算 因为只是前向传播获取隐藏层状态，所以不需要计算梯度
        last_hidden_states = (model(tokens_tensor)[1].squeeze(0)).tolist()
        de_bert.append(list(last_hidden_states))
de_bert = np.array(de_bert)

target_data = (de_data, de_bert)
with open(save_de, 'wb') as file_de:
    pickle.dump(target_data, file_de)
"""end"""

"""读取数据"""
# path = '/home/snchen/project/openkiwi-init-addsent/data_embeddings/exp3/en_word_embed.pkl'
# f = open(path, 'rb')
# data = pickle.load(f, encoding='latin1')
# print(data)
