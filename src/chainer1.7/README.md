
# インストール

python2.7<br>
chainer1.7<br>

	
	pip install chainer==1.7

# コンバート

    python convert_caffe_to_chainer.py ../../model/bvlc_alexnet.caffemodel ../../model/bvlc_reference_chainermodel.pkl

# caffemodelで実行


    python evaluate_caffe_net_ranking.py ../../data/img/banana.jpg caffenet bvlc_reference_chainermodel.pkl -b ./


# pklで実行


    python evaluate_caffe_net_ranking.py ../../data/img/banana.jpg caffenet bvlc_reference_chainermodel.pkl -b ./



