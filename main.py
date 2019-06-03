import urllib.request, urllib.error
import re
import json
import glob
import rating
import file_writer
import summarizer

def numericalSort(value):
    """
    番号順にソート.
    コーパスファイル読み込みに使用
    """
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

if __name__ == '__main__':
    """ 
    @param rank             上位何個まで抽出するかを決定する
    @param in_filename      入力ファイル名
    @param in_encode        コーパスファイルのエンコード. utf-8 or shift-jis
    @param out_nameorder    分かち書きした各単語の順番を保存するファイル名
    @param out_tf_idf       tf_idfの上位rank位までを出力するファイル名
    @param out_okapi        Okapi BM25の上位rank位までを出力するファイル名
    @param out_okapi_plus   Okapi BM25+の上位rank位までを出力するファイル名
    """
    rank = 5
    in_encode = "utf-8"
    out_documents = "./result/documents.json"
    out_nameorder = "./result/words_list.json"
    out_tf_idf = f"./result/tf_idf_top{rank}.json"
    out_okapi = f"./result/okapi_top{rank}.json"
    out_okapi_plus = f"./result/okapi_plus_top{rank}.json"

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
    files_path = sorted(glob.glob("corpus/Tsukiya*.txt"), key=numericalSort)
    for path in files_path:
        print(path)
        with open(path, "r", encoding=in_encode) as rfp:
            contents = rfp.read().replace('\n', '')
            document_list.append(contents)

    """
    ドキュメントをjsonにまとめて出力
    """
    with open(out_documents, "w") as cf:
        json.dump(document_list, cf, indent=2, ensure_ascii=False)

    """
    rating による処理
    """
    rating = rating.rating(slothlib_stopwords, normalize=True)
    tf_idf_sparse = rating.get_tf_idf(document_list)
    okapi_sparse = rating.get_okapi(document_list)
    okapi_plus_sparse = rating.get_okapi_plus(document_list)
    name = rating.get_words_list()

    """
    単語一覧をjsonに書き出し
    """
    with open(out_nameorder, "w") as out_1:
        json.dump(name, out_1, indent=2, ensure_ascii=False)
        for i in range(len(name)):
            print(i, name[i], file=out_1)

    """
    上からrank位までの単語をjsonに書き出し
    """
    tf_idf_high_rank = file_writer.write_words_of_high_rank(out_tf_idf, tf_idf_sparse, name, rank)
    okapi_high_rank = file_writer.write_words_of_high_rank(out_okapi, okapi_sparse, name, rank)
    okapi_plus_high_rank = file_writer.write_words_of_high_rank(out_okapi_plus, okapi_plus_sparse, name, rank)

    """
    high_rankの単語が含まれる文を抜き出すことで、要約を行う。
    """
    tf_idf_summarize = summarizer.summarizer(document_list, tf_idf_high_rank)
    okapi_summarize = summarizer.summarizer(document_list, okapi_high_rank)
    okapi_plus_summarize = summarizer.summarizer(document_list, okapi_plus_high_rank)

    """
    要約文書き出し
    """
    file_writer.write_summary_json("result/summary_tf_idf.json", tf_idf_summarize)
    file_writer.write_summary_json("result/summary_okapi.json", okapi_plus_summarize)    
    file_writer.write_summary_json("result/summary_okapi_plus.json", okapi_plus_summarize)

    file_writer.write_summary_individual("tf_idf", tf_idf_summarize)
    file_writer.write_summary_individual("okapi", okapi_summarize)
    file_writer.write_summary_individual("okapi_plus", okapi_plus_summarize)
    
