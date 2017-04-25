
# インストール

python2.7<br>
chainer1.7<br>

	
	pip install chainer==1.7


# 学習済みモデルと平均画像データのダウンロード

    python download_mean_file.py
    python download_model.py alexnet


# コンバート

    python convert_caffe_to_chainer.py ../../model/bvlc_alexnet.caffemodel ../../model/bvlc_reference_chainermodel.pkl

# caffemodelで実行


    python evaluate_caffe_net_ranking.py ../../data/img/banana.jpg caffenet ../../model/bvlc_reference_chainermodel.pkl -b ./


# pklで実行

    python evaluate_caffe_net_ranking.py ../../data/img/banana.jpg alexnet ../../model/bvlc_reference_chainermodel.pkl -b ./
    python evaluate_caffe_net_ranking.py ../../data/img/banana.jpg caffenet ../../model/bvlc_reference_chainermodel.pkl -b ./



