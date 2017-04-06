# fasttext sample

https://deepage.net/bigdata/machine_learning/2016/08/28/fast_text_facebook.html

# setup

    wget http://www.rondhuit.com/download/ldcc-20140209.tar.gz
    tar -zxvf ldcc-20140209.tar.gz
    python livedoor.py > livedoor.txt
    python rand_split.py

    git clone https://github.com/facebookresearch/fastText.git
    cd fastText
    make
   
# train

  ./fasttext supervised -input ../livedoor_train.txt -output livedoor_result -dim 10 -lr 0.1 -wordNgrams 2 -minCount 1 -bucket 10000000 -epoch 100 -thread 4


