import urllib.request, urllib.error
import MeCab
import Okapi

def extract_words(document, stopwords):

    # MeCab展開
    tagger = MeCab.Tagger('-Owakati -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
    words = tagger.parse(document)
    splited_words = words.split(' ')

    print(splited_words)

    # modified_words = [word for word in splited_words if word not in stopwords]

    # return modified_words
    return splited_words

if __name__ == '__main__':
    # ファイルをオープンする
    in_filename = "corpus20150426_with_titles.txt"
    out_filename = "output_name.txt"
    document_list = []

    slothlib_path = 'http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt'
    slothlib_file = urllib.request.urlopen(slothlib_path)
    slothlib_stopwords = [line.decode("utf-8").strip() for line in slothlib_file]
    slothlib_stopwords = [ss for ss in slothlib_stopwords if not ss==u'']

    with open(in_filename, "r", encoding="shift_jis") as file:
        # 一行ずつ読み込んでは表示する
        for line in file:
            document_list.append(line)
            if (line == '\n'):
                break

    f = extract_words

    okapi = Okapi.Okapi(f, slothlib_stopwords)
    res = okapi.fit_transform(document_list)
    name = okapi.get_feature_names()

    # print(res)

    with open(out_filename, "w") as out_file:
        for i in range(len(name)):
            print(i, name[i], file=out_file)
        # print(res, file=out_file)


