import re

def summarizer(document_list: str, high_rank_all: dict):
    """
    上位の単語を含む文を抽出する.

    Args:
        document_list (string): 文章リスト
        high_rank_all (dict): 上位の単語集合
    Returns:
        抽出された文章のリスト
    """
    summarize = []
    for document, high_rank in zip(document_list, high_rank_all):
        summarize_text = set()
        document = re.split("[ 。・\?\!？！■]", document)
        for sentence in document:
            for hr in list(high_rank.keys()):
                if hr in sentence:
                    summarize_text.add(sentence)
        summarize.append(list(summarize_text))

    return summarize