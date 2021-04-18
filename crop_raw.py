import os
from glob import glob
import h5py
import scipy.io as sio
import numpy as np
from tqdm import tqdm
import argparse
from PIL import Image
import csv


map_table = {
    'GP': 'bggr',
    'IP': 'rggb',
    'S6': 'grbg',
    'N6': 'bggr',
    'G4': 'bggr'
}



def raw2tensor(raw, bayer_pattern):
    if bayer_pattern.lower() == 'bggr':
        r = raw[1::2,1::2]
        g2 = raw[0::2,1::2]
        g1 = raw[1::2,0::2]
        b = raw[0::2,0::2]
    elif bayer_pattern.lower() == 'rggb':
        r = raw[0::2,0::2]
        g1 = raw[0::2,1::2]
        g2 = raw[1::2,0::2]
        b = raw[1::2,1::2]
    elif bayer_pattern.lower() == 'grbg':
        r = raw[0::2,1::2]
        g1 = raw[0::2,0::2]
        g2 = raw[1::2,1::2]
        b = raw[1::2,0::2]
    else:
        raise Exception("Sorry, this bayer pattern isnot processed")
    
    return np.stack([r,g1,g2,b], axis=-1)


parser = argparse.ArgumentParser(prog='SIDD Train dataset Generation')
parser.add_argument('--data_dir', default=None, type=str, metavar='PATH',
                                      help="path to save the training set of SIDD, (default: None)")
args = parser.parse_args()

path_all_noisy = glob(os.path.join(args.data_dir, '**/*NOISY*.MAT'), recursive=True)
path_all_noisy = sorted(path_all_noisy)
path_all_gt = [x.replace('NOISY', 'GT') for x in path_all_noisy]
print('Number of big images: {:d}'.format(len(path_all_gt)))

print('Training: Split the original images to small ones!')

pch_size = 512
stride = 512-128
num_patch = 0

parent_dir = 'crop_medium'
try:
    os.mkdir(parent_dir)
except: pass

NLF = {}
with open(os.path.join(args.data_dir, 'noise_level_functions.csv')) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    header = csv_file.readline().replace('\n','')
    for row in csv_file.readlines():
        row = row.replace('\n','')
        items = row.split(',')
        NLF[items[0]] = {'beta1_r': float(items[1]),
                         'beta2_r': float(items[2]),
                         'beta1_g': float(items[3]),
                         'beta2_g': float(items[4]),
                         'beta1_b': float(items[5]),
                         'beta2_b': float(items[6])}


for ii in tqdm(range(len(path_all_gt))):

    folder_path = os.path.split(path_all_noisy[ii])[0]
    folder = os.path.split(folder_path)[-1]
    
    device = folder[9:11]
    pattern = map_table[device]

    noisy_h5 = h5py.File(path_all_noisy[ii], 'r')
    gt_h5 = h5py.File(path_all_gt[ii], 'r')

    im_noisy_int8 = raw2tensor(noisy_h5['x'][...].T, pattern)
    im_gt_int8 = raw2tensor(gt_h5['x'][...].T, pattern)
    
    H, W = im_noisy_int8.shape[:2]
    ind_H = list(range(0, H-pch_size+1, stride))
    if ind_H[-1] < H-pch_size:
        ind_H.append(H-pch_size)
    ind_W = list(range(0, W-pch_size+1, stride))
    if ind_W[-1] < W-pch_size:
        ind_W.append(W-pch_size)
    for start_H in ind_H:
        for start_W in ind_W:
            pch_noisy = im_noisy_int8[start_H:start_H+pch_size, start_W:start_W+pch_size]
            pch_gt = im_gt_int8[start_H:start_H+pch_size, start_W:start_W+pch_size]

            folder_path = os.path.join(parent_dir, str(num_patch)) 
            # try:
            if not os.path.isdir(folder_path):
                os.mkdir(folder_path)
                # except: pass
                print(folder_path)
            sio.savemat(os.path.join(folder_path, 'noisy.mat'), {**{'x':pch_noisy}, **NLF[folder]})
            sio.savemat(os.path.join(folder_path, 'clean.mat'), {**{'x':pch_gt}, **NLF[folder]})
            num_patch += 1

    noisy_h5.close()
    gt_h5.close()

        
print('Total {:d} small images in training set'.format(num_patch))
print('Finish!\n')

