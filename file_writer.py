import json

def write_words_of_high_rank(filename, sparse_matrix, word_name, num):
    """ 
    上位 num 件の単語を抜き出し、{method}_top{num}.jsonに書き込む.

    Args:
        filename: 保存する.jsonのファイル名
        sparse_matrix: rating結果が格納された疎行列
        word_name: インデックスと単語の対応が格納された辞書
        num: 抽出する件数(上位 num 個)
    Returns:
        上位の単語リスト
    """
    # 疎行列を展開
    matrix = sparse_matrix.toarray()
    
    high_rank_array = []
    file_num = 0

    for doc_weight in matrix:
        # 各文章ごとに直す
        words_weight = dict(zip(word_name, doc_weight))
        i = 0

        high_rank = {}
        # 上位を抜き出し
        for k, v in sorted(words_weight.items(), key=lambda x: -x[1]):
            high_rank[k] = v

            i += 1
            if i >= num:
                break

        high_rank_array.append(high_rank)

        # jsonで書き込み
    with open(filename, "w") as wfp:
        json.dump(high_rank_array, wfp, indent=2, ensure_ascii=False)

    return high_rank_array

def write_summary_json(filename, summerize_arr):
    """
    要約結果をjsonファイルに書き出す.

    Args:
        filename: 書き出すファイル名
        summerize_arr: 要約文章のリスト
    """

    with open(filename, "w") as wfp:
        json.dump(summerize_arr, wfp, indent=2, ensure_ascii=False)

def write_summary_individual(method: str, summerize_arr):
    """
    要約結果を個別のtxtに書き出す.

    Args:
        method: 使用した手法 (tf-idf, okapi, okapi_plus)
        summerize_arr: 要約文章のリスト
    """

    i = 0

    for summerize_doc in summerize_arr:
        filename = "./result/summary_" + method + "/" + method + "_" + str(i) + ".txt"
        with open(filename, "w") as wfp:
            wfp.write('\n'.join(summerize_doc))
        i += 1