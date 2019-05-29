import numpy as np
from scipy import sparse as sp
import math


class Okapi:
    def __init__(self, extract_words_func: callable, stopwords, b=0.75, k1=2.0, delta=1.0, norm=True):
        """
        :param extract_words_func: ドキュメントを単語リスト化する関数オブジェクト。引数にstopwordsが必要。
        :param stopwords: ストップワードリスト
        :param b: constant
        :param k1: constant
        :param delta: constant
        """
        self.K1, self.B, self.delta = k1, b, delta  # 定数
        self.norm = norm  # 正規化するかしないか
        self.word_index_dict, self.counter = {}, 0  # 単語とインデックスの辞書, 単語数
        self.feature_names = []  # 単語名のリスト
        self.idf = np.array([])  # inverse document frequency
        self.average_words_len = 0  # ドキュメント内の単語数の平均
        self.stopwords = stopwords  # ストップワード
        if callable(extract_words_func):
            self.extract_words_func = extract_words_func
        else:
            raise RuntimeError("extract_words_funcは呼び出し可能オブジェクトでなければいけません")

    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)

    def fit(self, documents):
        """
        ベクトライザーのセットアップ
        IDFのみ設定
        :param documents:
        """
        for document in documents:
            searched_dict = {}
            word_ls = self.extract_words_func(document, self.stopwords)
            self.average_words_len += len(word_ls)
            for word in word_ls:
                # このドキュメント内で始めて出た単語
                if not searched_dict.get(word):
                    searched_dict[word] = True
                    # 他のドキュメントですでに出た単語
                    if self.word_index_dict.get(word):
                        self.idf[self.word_index_dict[word]] += 1.0
                    # 初めて出現する単語
                    else:
                        self.feature_names.append(word)
                        self.word_index_dict[word] = self.counter
                        self.counter += 1
                        self.idf = np.append(self.idf, [1.0])

        print(self.idf)
        documents_len = len(documents)
        self.idf = np.log2((documents_len-self.idf+0.5) / (self.idf+0.5))
        self.average_words_len = self.average_words_len / documents_len


    def transform(self, documents):
        """
        ドキュメントを重み付け
        :param documents:
        :return: scipy.sparse.lil_matrixオブジェクト
        """
        len_ls = []  # 1つのドキュメントに出現した単語数
        word_index_ls, word_weight_ls = [], []  # 単語のインデックス, 単語の出現数
        for doc in documents:
            word_weight_dict = {}
            word_ls = self.extract_words_func(doc, self.stopwords)
            # Term Frequency
            for word in word_ls:
                if self.word_index_dict.get(word):
                    if word_weight_dict.get(self.word_index_dict[word]):
                        word_weight_dict[self.word_index_dict[word]] += 1.0
                    else:
                        word_weight_dict[self.word_index_dict[word]] = 1.0

            # Convine Weigth重み付け
            dist = 0
            word_len = len(word_ls)
            for ind in word_weight_dict.keys():
                word_weight_dict[ind] = self.idf[ind] * \
                                        (self.delta + (word_weight_dict[ind] * (self.K1+1.0)) /\
                                    (word_weight_dict[ind] + self.K1 * (1.0 - self.B + self.B*(word_len / self.average_words_len))))
                dist += word_weight_dict[ind] ** 2
            if self.norm:
                # 正規化
                dist = math.sqrt(dist)
                for ind in word_weight_dict.keys():
                    word_weight_dict[ind] /= dist

            len_ls.append(len(word_weight_dict))
            word_index_ls.extend(word_weight_dict.keys())
            word_weight_ls.extend(word_weight_dict.values())

        # sp.lil_matrixで疎行列オブジェクト生成
        it = 0
        result = sp.lil_matrix((len(documents), self.counter))
        for i in range(len(len_ls)):
            for _ in range(len_ls[i]):
                result[i, word_index_ls[it]] = word_weight_ls[it]
                it += 1
        return result

    def get_feature_names(self):
        return self.feature_names
