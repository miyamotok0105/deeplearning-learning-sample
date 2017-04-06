# -*- coding: utf-8 -*-
import MeCab
import os
from os import listdir, walk
from os.path import join, isfile

mecab = MeCab.Tagger('mecabrc')

labels = [
    'dokujo-tsushin',
    'it-life-hack',
    'kaden-channel',
    'livedoor-homme',
    'movie-enter',
    'peachy',
    'smax',
    'sports-watch',
    'topic-news'
]

def tokenize(text):
    node = mecab.parseToNode(text.strip())
    tokens = []
    while node:
        tokens.append(node.surface)
        node = node.next
    return ' '.join(tokens)

def is_post(directory, filename):
    if isfile(join(directory, filename)):
        if filename != 'LICENSE.txt':
            return True
    return False

def read_content(directory, filename):
    body = [l.strip() for i, l in enumerate(open(join(directory, filename), 'r')) if i > 1]
    text = ''.join(body)
    return tokenize(text)

def read_posts(directory):
    if os.path.exists(join(directory, 'LICENSE.txt')):
        files = [f for f in listdir(directory) if is_post(directory, f)]
        for f in files:
            yield read_content(directory, f)

def read_corpus():
    for (dirpath, _, _) in walk('text/'):
        label_name = os.path.basename(dirpath)
        for post in read_posts(dirpath):
            print('__label__{} , {}'.format(labels.index(label_name), post))

if __name__ == '__main__':
    read_corpus()
