import urllib.request, urllib.error
import json
import Summarizer
import csv
import numpy as np
import pathlib as Path

def write_file(filename, sparse_matrix, word_name, num):
    matrix = sparse_matrix.toarray()
    weight = matrix.sum(axis=0)
    words_weight = dict(zip(word_name, weight))
    i = 0

    # 上位を抜き出し
    over25 = {}
    for k, v in sorted(words_weight.items(), key=lambda x: -x[1]):
        over25[k] = v

        i += 1
        if i >= num:
            break

    # jsonで書き込み
    with open(filename, "w") as wfp:
        json.dump(over25, wfp, indent=2, ensure_ascii=False)




if __name__ == '__main__':
    """ 
    @param in_filename      入力ファイル名
    @param in_encode        utf-8 or shift-jis
    @param out_nameorder    分かち書きした各単語の順番を保存するtxt
    @param 
    """
    in_encode = "utf-8"
    out_documents = "./result/documents.json"
    out_nameorder = "./result/words_list.json"
    out_tf_idf = "./result/summarize_tf_idf_over25.json"
    out_okapi = "./result/summarize_okapi_over25.json"
    out_okapi_plus = "./result/summarize_okapi_plus_over25.json"

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
    dir_path = Path.Path('corpus')
    files_path = dir_path.glob("*.txt")
    for path in files_path:
        with open(path, "r", encoding=in_encode) as rfp:
            contents = rfp.read().replace('\n', '')
            document_list.append(contents)

    # デバッグ用
    with open(out_documents, "w") as cf:
        json.dump(document_list, cf, indent=2, ensure_ascii=False)

    """
    Okapi による処理
    """
    summarizer = Summarizer.Summarizer(slothlib_stopwords, normalize=True)
    tf_idf_sparse = summarizer.get_tf_idf(document_list)
    okapi_sparse = summarizer.get_okapi(document_list)
    okapi_plus_sparse = summarizer.get_okapi_plus(document_list)
    name = summarizer.get_words_list()

    """
    ファイル書き込み
    """
    with open(out_nameorder, "w") as out_1:
        json.dump(name, out_1, indent=2, ensure_ascii=False)
        # for i in range(len(name)):
        #     print(i, name[i], file=out_1)

    with open(out_tf_idf, "w") as fp:
        fp.write(str(tf_idf_sparse))

    with open(out_okapi, "w") as fp2:
        fp2.write(str(okapi_sparse))

    with open(out_okapi_plus, "w") as fp3:
        fp3.write(str(okapi_plus_sparse))

    """
    各単語の重要度を加算、上から数件表示
    """
    write_file(out_tf_idf, tf_idf_sparse, name, 25)
    write_file(out_okapi, okapi_sparse, name, 25)
    write_file(out_okapi_plus, okapi_plus_sparse, name, 25)