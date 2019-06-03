# 自然言語処理 E班 成果物

## プログラム構成
- main.py: メイン関数
- rating.py: tf-idf, okapi, okapi bm25, okapi bm25+を実行するスクリプト
- summarizer.py: ratingの結果から、文章の要約を行うスクリプト
- file_writer.py: ファイルの書き出しを行うスクリプト

## 入力
corpus ディレクトリの中に入っているすべての.txtファイル

## 出力
result ディレクトリについて
- documents.json: 読み込んだtxtファイルをつなげて出力しただけ
- words_list.json: 分かち書きした単語リスト。今回はMeCabを利用して形態素解析し、助詞, 助動詞, 接続詞, 記号, 動詞は削除している
- {手法名}_top5.json: 順位付け手法で得られた値のうち、各文章ごとに値が大きかった上位5単語を出力したjson
- summary_{手法名}.json: 要約文すべてを出力したもの

summary_tf-idf, summary_okapi, summary_okapi_plus には、summary_{手法名}.json を各txtファイルごとに分けた物が入っています。
0番から生成してしまったので、corpus の Tsukiya1.txt の要約結果は、tf_idf_0.txt, okapi_0.txt, okapi_plus_0.txt に入っています。
