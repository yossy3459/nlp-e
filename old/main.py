import urllib.request, urllib.error
import MeCab
import json
import numpy as np
import Okapi

def extract_words(document, stopwords):
    """
    MeCabによる分かち書き + ストップワード削除.
    助詞、助動詞、接続詞、記号 + 引数で渡されたストップワードを削除
    tf-idfやOkapiの処理クラスに渡すコールバック関数

    @param document 分かち書きしたい文章
    @param stopwords ストップワード、別途指定された取り除きたい単語のこと
    """
    # Mecab処理
    tagger = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
    splited_words = [x.split('\t')[0] for x in tagger.parse(document).splitlines()[:-1] if x.split('\t')[1].split(',')[0] not in ['助詞', '助動詞', '接続詞', '記号']]

    # ストップワード除去
    modified_words = [word for word in splited_words if word not in stopwords]

    return modified_words


if __name__ == '__main__':
    """ 
    @param in_filename      入力ファイル名
    @param in_encode        utf-8 or shift-jis
    @param out_nameorder    分かち書きした各単語の順番を保存するtxt
    @param 
    """
    in_filename = "corpus.txt"
    in_encode = "utf-8"
    out_nameorder = "wakachigaki_name_order.json"
    out_okapi = "result_okapi.txt"

    """
    ストップワードをリスト化
    """
    slothlib_path = 'http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt'
    slothlib_file = urllib.request.urlopen(slothlib_path)
    slothlib_stopwords = [line.decode("utf-8").strip() for line in slothlib_file]
    slothlib_stopwords = [w for w in slothlib_stopwords if not w==u'']

    """
    コーパスファイルオープン
    """
    document_list = []
    with open(in_filename, "r", encoding=in_encode) as file:
        # 一行ずつ読み込んでは表示する
        for line in file:
            document_list.append(line)
            if (line == '\n'):
                break

    """
    コールバック関数
    """
    f = extract_words

    """
    Okapi による処理
    """
    okapi = Okapi.Okapi(f, slothlib_stopwords)
    res = okapi.fit_transform(document_list)
    name = okapi.get_feature_names()

    """
    ファイル書き込み
    """
    with open(out_nameorder, "w") as out_1:
        json.dump(name, out_1, indent=2, ensure_ascii=False)
        # for i in range(len(name)):
        #     print(i, name[i], file=out_1)

    with open(out_okapi, "w") as out_2:
        out_2.write(str(res))


