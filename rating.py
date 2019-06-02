import numpy as np
from scipy import sparse as sp
import math
import MeCab

class rating:
    def __init__(self, stopwords, b=0.75, k1=2.0, normalize=False):
        """
        コンストラクタ.

        Args:
            stopwords: ストップワード、別途指定された取り除きたい単語のこと
            b, k1, delta: Okapi BM25 における定数
            norm: 正規化するか
        """
        self.stopwords = stopwords
        self.b, self.k1 = b, k1
        self.normalize = normalize

        self.word_id = {}  # Key=単語, Value=管理番号
        self.words_count = 0
        self.words_len = np.array([])  # 各文章の単語数
        self.documents_len = 0  # 文章数
        self.words_list = []  # 全単語リスト

        self.df_count = np.array([])  # 出現回数

    def reset(self):
        """
        リセット用.
        前回のデータを消去する
        """
        self.word_id = {}  # Key=単語, Value=管理番号
        self.words_count = 0
        self.words_len = np.array([])  # 各文章の単語数
        self.documents_len = 0  # 文章数
        self.words_list = []  # 全単語リスト

        self.df_count = np.array([])  # 出現回数

    def get_tf_idf(self, documents):
        """
        TF-IDF値を取得する. main用

        Args:
            document: 文章
        Returns:
            各文章・各単語ごとのTF-IDFの値が格納された疎行列
        """
        self.reset()
        self.setup_vectorizer(documents)
        return self.transform(documents, "tf-idf")

    def get_okapi(self, documents):
        """
        Okapi BM25値を取得する. main用

        Args:
            document: 文章
        Returns:
            各文章・各単語ごとのOkapi BM25の値が格納された疎行列
        """
        self.reset()
        self.setup_vectorizer(documents)
        return self.transform(documents, "okapi")

    def get_okapi_plus(self, documents):
        """
        Okapi BM25+値を取得する. 必ず正規化を行う main用

        Args:
            document: 文章
        Returns:
            各文章・各単語ごとのOkapi BM25+の値が格納された疎行列
        """
        self.reset()
        self.setup_vectorizer(documents)
        return self.transform(documents, "okapi-plus")

    

    def extractor(self, document):
        """
        MeCabによる分かち書き + ストップワード削除.
        助詞、助動詞、接続詞、記号 + 引数で渡されたストップワードを削除
        tf-idfやOkapiの処理クラスに渡すコールバック関数

        Args:
            document: 分かち書きしたい文章
        Returns:
            分かち書きされた単語のリスト
        """
        # Mecab処理
        tagger = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
        splited_words = [x.split('\t')[0] for x in tagger.parse(document).splitlines()[:-1] if x.split('\t')[1].split(',')[0] not in ['助詞', '助動詞', '接続詞', '記号', '動詞']]

        # ストップワード除去
        modified_words = [word for word in splited_words if word not in self.stopwords]

        return modified_words

    def setup_vectorizer(self, documents):
        """
        ドキュメントや文章の長さ、単語の出現回数などの情報を抽出

        Args:
            document: コーパスの文章
        """
        self.documents_len = len(documents)

        for document in documents:
            word_exists_in_doc = {}  # Key=単語、Value=T or F
            extracted_words = self.extractor(document)
            self.words_len = np.append(self.words_len, len(extracted_words))

            for word in extracted_words:
                # このdocument中で初めて出たなら、word_existsに登録
                if not word_exists_in_doc.get(word):
                    word_exists_in_doc[word] = True
                    # 他で出ているなら, df_count += 1
                    if self.word_id.get(word):
                        self.df_count[self.word_id[word]] += 1.0
                    # 初めてなら、words_list, word_idに追加, df_countの末尾に1追加
                    else:
                        self.words_list.append(word)
                        self.word_id[word] = self.words_count
                        self.words_count += 1
                        self.df_count = np.append(self.df_count, 1.0)

    def transform(self, documents, mode):
        """
        各種情報から, 順位付け手法による値を得る.

        Args:
            document: コーパスの文章
            mode (string): 'tf-idf', 'okapi', 'okapi-plus'のいずれか. それ以外はエラー
        Returns:
            各文章・各単語ごとの順位付け手法の結果 (疎行列 sparse_matlix)
        """
        # エラー処理
        if not (mode == "tf-idf" or mode == "okapi" or mode == "okapi-plus"):
            raise RuntimeError("mode は 'tf-idf', 'okapi', 'okapi-plus' のいずれかを指定してください。")

        # 結果格納用
        result_keys, result_values, result_len = [], [], []

        # Okapi用avgdl, 詳しくはOkapiのWikipedia参照
        avgdl = np.average(self.words_len)

        # idf値の算出
        if mode == 'tf-idf':
            idf = np.log2(self.documents_len / (self.df_count + 1))  # ゼロ除算を防ぐため、分母に+1
        else:
            idf = np.log2((self.documents_len - self.df_count + 0.5) / (self.df_count + 0.5))

        # 各ドキュメントに対する処理を行う
        for document in documents:
            # 文ごとの結果
            result_doc = {}

            # 文を展開
            extracted_words = self.extractor(document)
            extracted_words_len = len(extracted_words)

            # 文ごとの単語出現回数を計測
            for word in extracted_words:
                if self.word_id.get(word):
                    if result_doc.get(self.word_id[word]):
                        result_doc[self.word_id[word]] += 1.0
                    else:
                        result_doc[self.word_id[word]] = 1.0
            
            # 正規化用のノルムの値
            norm = 0

            # 各種値の計算
            for i in result_doc.keys():
                
                if mode == 'tf-idf':
                    result_doc[i] = (result_doc[i] / extracted_words_len) * idf[i]

                if mode == 'okapi':
                    result_doc[i] = idf[i] * ((result_doc[i] * (self.k1 + 1.0)) /\
                                (result_doc[i] + self.k1 * (1.0 - self.b + self.b * (extracted_words_len / avgdl))))

                if mode == 'okapi-plus':
                    result_doc[i] = idf[i] * (1.0 + (result_doc[i] * (self.k1 + 1.0)) /\
                                (result_doc[i] + self.k1 * (1.0 - self.b + self.b * (extracted_words_len / avgdl))))

                norm += result_doc[i] ** 2

                

            # 正規化
            if (mode != 'okapi-plus' and self.normalize) or mode == 'okapi-plus':
                norm = math.sqrt(norm)
                for i in result_doc.keys():
                    result_doc[i] /= norm

            # 文ごとに得た値をリストに追加、文章にまとめる
            result_len.append(len(result_doc))
            result_keys.extend(result_doc.keys())
            result_values.extend(result_doc.values())

        # 疎行列で出力
        it = 0
        result = sp.lil_matrix((self.documents_len, self.words_count))
        for i in range(len(result_len)):
            for _ in range(result_len[i]):
                result[i, result_keys[it]] = result_values[it]
                it += 1
        
        return result

    def get_words_list(self):
        """
        単語とインデックスのリストを取得.

        Returns:
            単語リスト
        """
        return self.words_list








