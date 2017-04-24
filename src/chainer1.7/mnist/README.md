

http://y-uti.hatenablog.jp/entry/2014/07/23/074845

# データの展開

    gzip -dc train-images-idx3-ubyte.gz >train-images-idx3-ubyte
    gzip -dc train-labels-idx1-ubyte.gz >train-labels-idx1-ubyte

# データの変換

    od -An -v -tu1 -j16 -w784 train-images-idx3-ubyte | sed 's/^ *//' | tr -s ' ' >train-images.txt
    od -An -v -tu1 -j8 -w1 train-labels-idx1-ubyte | tr -d ' ' >train-labels.txt


