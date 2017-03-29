

#コンバード

    python convert_caffe_to_chainer.py bvlc_reference_caffenet.caffemodel bvlc_reference_chainermodel.pkl


#caffemodelで実行


    python evaluate_caffe_net_ranking.py ../../data/img/banana.jpg caffenet bvlc_reference_chainermodel.pkl -b ./


#pklで実行


    python evaluate_caffe_net_ranking.py ../../data/img/banana.jpg caffenet bvlc_reference_chainermodel.pkl -b ./



