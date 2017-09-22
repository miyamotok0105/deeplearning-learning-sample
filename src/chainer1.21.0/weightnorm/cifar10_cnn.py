#coding: utf-8
import argparse
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.datasets import get_cifar10
from chainer.datasets import get_cifar100

class Block(chainer.Chain):
    def __init__(self, out_channels, ksize, pad=1):
        super(Block, self).__init__(
            conv=L.Convolution2D(None, out_channels, ksize, pad=pad,
                                 nobias=True),
            bn=L.BatchNormalization(out_channels)
        )

    def __call__(self, x, train=True):
        h = self.conv(x)
        h = self.bn(h, test=not train)
        return F.relu(h)

class VGG(chainer.Chain):
    def __init__(self, class_labels=10):
        super(VGG, self).__init__(
            block1_1=Block(64, 3),
            block1_2=Block(64, 3),
            block2_1=Block(128, 3),
            block2_2=Block(128, 3),
            block3_1=Block(256, 3),
            block3_2=Block(256, 3),
            block3_3=Block(256, 3),
            block4_1=Block(512, 3),
            block4_2=Block(512, 3),
            block4_3=Block(512, 3),
            block5_1=Block(512, 3),
            block5_2=Block(512, 3),
            block5_3=Block(512, 3),
            fc1=L.Linear(None, 512, nobias=True),
            bn_fc1=L.BatchNormalization(512),
            fc2=L.Linear(None, class_labels, nobias=True)
        )
        self.train = True

    def __call__(self, x):
        # 64 channel blocks:
        h = self.block1_1(x, self.train)
        h = F.dropout(h, ratio=0.3, train=self.train)
        h = self.block1_2(h, self.train)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # 128 channel blocks:
        h = self.block2_1(h, self.train)
        h = F.dropout(h, ratio=0.4, train=self.train)
        h = self.block2_2(h, self.train)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # 256 channel blocks:
        h = self.block3_1(h, self.train)
        h = F.dropout(h, ratio=0.4, train=self.train)
        h = self.block3_2(h, self.train)
        h = F.dropout(h, ratio=0.4, train=self.train)
        h = self.block3_3(h, self.train)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # 512 channel blocks:
        h = self.block4_1(h, self.train)
        h = F.dropout(h, ratio=0.4, train=self.train)
        h = self.block4_2(h, self.train)
        h = F.dropout(h, ratio=0.4, train=self.train)
        h = self.block4_3(h, self.train)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # 512 channel blocks:
        h = self.block5_1(h, self.train)
        h = F.dropout(h, ratio=0.4, train=self.train)
        h = self.block5_2(h, self.train)
        h = F.dropout(h, ratio=0.4, train=self.train)
        h = self.block5_3(h, self.train)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = F.dropout(h, ratio=0.5, train=self.train)
        h = self.fc1(h)
        h = self.bn_fc1(h, test=not self.train)
        h = F.relu(h)
        h = F.dropout(h, ratio=0.5, train=self.train)
        return self.fc2(h)

class TestModeEvaluator(extensions.Evaluator):

    def evaluate(self):
        model = self.get_target('main')
        model.train = False
        ret = super(TestModeEvaluator, self).evaluate()
        model.train = True
        return ret

def main():
    parser = argparse.ArgumentParser(description='Chainer CIFAR example:')
    parser.add_argument('--dataset', '-d', default='cifar10',
                        help='The dataset to use: cifar10 or cifar100')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=300,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')
    if args.dataset == 'cifar10':
        print('Using CIFAR10 dataset.')
        class_labels = 10
        train, test = get_cifar10()
    elif args.dataset == 'cifar100':
        print('Using CIFAR100 dataset.')
        class_labels = 100
        train, test = get_cifar100()
    else:
        raise RuntimeError('Invalid dataset choice.')
    model = L.Classifier(VGG(class_labels))
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU
    optimizer = chainer.optimizers.MomentumSGD(0.1)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(TestModeEvaluator(test_iter, model, device=args.gpu))
    trainer.extend(extensions.ExponentialShift('lr', 0.5),
                   trigger=(25, 'epoch'))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)
    trainer.run()

if __name__ == '__main__':
    main()
