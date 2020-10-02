import numpy as np
import pickle as pkl


def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    余弦值越接近1，就表明夹角越接近0度，也就是两个向量越相似
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    sim = num / denom
    return sim


vocab1 = 'school'
vocab2 = 'red'

"""start:polyglot"""
path = '/home/snchen/project/openkiwi-init-addsent/data_polyglot/embeddings2/en/words_embeddings_32.pkl'

f = open(path, 'rb')
content = pkl.load(f, encoding='latin1')
vocab = content[0]
embed = content[1]
# print(vocab)

index1 = [i for i, item in enumerate(vocab) if item == vocab1][0]
index2 = [i for i, item in enumerate(vocab) if item == vocab2][0]
embed1 = embed[index1]
embed2 = embed[index2]
diff = cos_sim(embed1, embed2)

print("polyglot提供的embedding中'{}'和'{}'这两个单词的index分别为：{}和{}，余弦相似度为：{}".format(vocab1, vocab2, index1, index2, diff))
# print("polyglot提供的embedding中'{}'和'{}'这两个单词的向量分别为：{}和{}".format(vocab1, vocab2, embed1, embed2))
"""end"""

"""start:bert_embedding"""
path = '/home/snchen/project/openkiwi-init-addsent/data_embeddings/exp1/en_word_embed.pkl'

f = open(path, 'rb')
content = pkl.load(f, encoding='latin1')
vocab = content[0]
embed = content[1]
# print(vocab)

index1 = [i for i, item in enumerate(vocab) if item == vocab1][0]
index2 = [i for i, item in enumerate(vocab) if item == vocab2][0]
embed1 = embed[index1]
embed2 = embed[index2]
diff = cos_sim(embed1, embed2)

print("transformers生成的embedding中'{}'和'{}'这两个单词的index分别为：{}和{}，余弦相似度为：{}".format(vocab1, vocab2, index1, index2, diff))
"""emd"""
