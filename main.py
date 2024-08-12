import jieba
import numpy as np
from numpy.linalg import norm


class WordVector:
    def __init__(self):
        # 词嵌入矩阵下载链接：https://ai.tencent.com/ailab/nlp/en/download.html
        self.fasttext_model = self.load_fasttext_model('model/tencent-ailab-embedding-zh-d100-v0.2.0-s.txt')

    def load_fasttext_model(self, filepath):
        word_vectors = {}
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.strip().split()
                word = values[0]
                vector = np.array(values[1:], dtype='float32')
                word_vectors[word] = vector
        return word_vectors

    def tokenize_text(self, text):
        return jieba.lcut(text)

    def get_sentence_vector(self, sentence):
        word_vectors = []
        vector_dim = len(next(iter(self.fasttext_model.values())))  # 获取词向量的维度

        for word in sentence:
            if word in self.fasttext_model:
                word_vectors.append(self.fasttext_model[word])
            else:
                # 如果词不在 fasttext_model 中，生成一个与词向量维度一致的随机向量
                random_vector = np.random.random(vector_dim)
                word_vectors.append(random_vector)
        # 如果没有任何词向量，返回全零向量
        if len(word_vectors) == 0:
            return np.zeros(vector_dim)
        return np.mean(word_vectors, axis=0)

    def get_text_vector(self, text):
        tokens = self.tokenize_text(text)
        return self.get_sentence_vector(tokens)

    @staticmethod
    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

    def are_sentences_similar(self, sentence1, sentence2, threshold=0.7):  # 调低阈值匹配更多相似
        vec1 = self.get_text_vector(sentence1)
        vec2 = self.get_text_vector(sentence2)
        similarity = self.cosine_similarity(vec1, vec2)
        return similarity >= threshold, similarity

    def compare_all_sentences(self, sentences):
        n = len(sentences)
        for i in range(n):
            for j in range(i + 1, n):
                is_similar, similarity_score = self.are_sentences_similar(sentences[i], sentences[j])
                print(f"'{sentences[i]}' 和 '{sentences[j]}' 的相似度: {similarity_score:.4f}，是否相似: {'是' if is_similar else '否'}")


if __name__ == '__main__':
    wv = WordVector()
    # 示例文本
    sentences = ["城市", "城市", "市", "城镇", "市区", "城市A名称", "主运营商城市", "纬度", "地理纬度", "地纬"]
    sentences2 = ["城市", "标识特定的电路或服务连接", "用户号码的区号", "请求的电话号码", "请求的项目地点", "所在的城市名称", "用户所在的城市名称", "用户所在的市区名", "用户所在的城市", "城市名"]
    wv.compare_all_sentences(sentences2)
