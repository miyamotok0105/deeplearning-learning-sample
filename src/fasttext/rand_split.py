import numpy as np
import os
from os.path import join

def append_newline(filename, line):
    with open(filename, 'a') as file:
        file.write(line)

def split_random(filename):
    dirname = os.path.dirname(filename)
    comps = os.path.basename(filename).split('.')
    train_name = '{}_train.{}'.format(comps[0], comps[1])
    test_name = '{}_test.{}'.format(comps[0], comps[1])
    for l in open(filename, 'r'):
        choice = np.random.choice([True, False], p=[0.9, 0.1])
        if choice: # write train
            append_newline(join(dirname, train_name), l)
        else: # write test
            append_newline(join(dirname, test_name), l)

if __name__ == '__main__':
    split_random('livedoor.txt')

