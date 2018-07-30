"""
Script for processing Few-Shot CIFAR dataset

Acquire CIFAR-100 dataset from http://www.cs.toronto.edu/~kriz/cifar.html
***Download CIFAR-100 and NOT CIFAR-10***
Extract cifar-100-python to current folder (data/cifar/)
Then run this script from the current directory (data/cifar/):
    python proc_images.py
"""

from __future__ import print_function

import os
import glob
import pickle
import numpy as np
from PIL import Image

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

datafiles = ['test', 'train']

labels = [label.decode('utf-8') for label in unpickle('cifar-100-python/meta')[b'fine_label_names']]

if not os.path.exists('images/'):
    os.makedirs('images/')

for label in labels:
    if not os.path.exists('images/' + label):
        os.makedirs('images/' + label + '/')

for datafile in datafiles:
    print("Extracting {} images...".format(datafile))
    data_dict = unpickle('cifar-100-python/' + datafile)
    for i, image in enumerate(data_dict[b'data']):
        label = labels[int(data_dict[b'fine_labels'][i])]
        new_path = 'images/' + label + '/' + data_dict[b'filenames'][i].decode('utf-8')
        Image.fromarray(np.transpose(image.reshape(3, 32, 32), axes=[1,2,0])).save(new_path)
        if (i + 1) % 5000 == 0:
            print("Extracted {} images...".format(i + 1))

print("Extracted images from pickle files...")
print("Begin assigning images to train/val/test folders...")

path_to_images = 'images/'

# Put in correct directory
for datatype in ['train', 'val', 'test']:
    if not os.path.exists(datatype):
        os.makedirs(datatype)

    with open(datatype + '.txt', 'r') as file:
        for line in file:
            label = line.rstrip('\n')
            new_dir = datatype + '/' + label + '/'
            old_dir = path_to_images + label + '/'
            os.makedirs(new_dir)
            filenames = glob.glob(path_to_images + label + '/*.png')
            for old_path in filenames:
                old_path = old_path.replace('\\', '/')
                new_path = new_dir + old_path.split('/')[-1].split('\\')[-1]
                os.rename(old_path, new_path)

print("Completed!")
