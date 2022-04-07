import os
import math
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import hflip
from PIL import Image

from model import ResNetFace


def parse_list(pairs_file, base_dir):
    pairs = []
    num_valid_data = 0
    with open(pairs_file) as lines:
        for line in lines:
            strings = line.split('\t')
            if len(strings) == 3:
                name, id1, id2 = strings
                file_l = os.path.join(base_dir, name, f'{name}_{int(id1):04}.jpg')
                file_r = os.path.join(base_dir, name, f'{name}_{int(id2):04}.jpg')
                flag = 1
            elif len(strings) == 4:
                name1, id1, name2, id2 = strings
                file_l = os.path.join(base_dir, name1, f'{name1}_{int(id1):04}.jpg')
                file_r = os.path.join(base_dir, name2, f'{name2}_{int(id2):04}.jpg')
                flag = -1
            else:
                continue

            if os.path.exists(file_l) and os.path.exists(file_r):
                num_valid_data += 1
                fold = math.ceil(num_valid_data / 600)
                pairs.append({
                        'file_l': file_l,
                        'file_r': file_r,
                        'flag': flag,
                        'fold': fold
                        })
    return pairs


def extract_deep_feature(files, model, device, batch_size):
    features = []
    for i in range(0, len(files), batch_size):
        images = [Image.open(f) for f in files[i:i+batch_size]]
        images = [np.array(i, dtype=np.float32).transpose(2, 0, 1) for i in images]
        images = (np.stack(images) - 127.5) / 128
        images = torch.tensor(images, device=device)

        img_feat = model(images)
        flipped_feat = model(hflip(images))
        feature = torch.hstack((img_feat, flipped_feat))
        features.append(feature)
    return torch.vstack(features)


def get_threshold(scores, flags, thr_num):
    thresholds = torch.arange(-1, 1, 1./thr_num)
    best_th, best_acc = -1.1, 0.
    for th in thresholds:
        acc = get_accuracy(scores, flags, th)
        if acc > best_acc:
            best_acc = acc
            best_th = th
    return best_th

def get_accuracy(scores, flags, threshold):
    flags = torch.tensor(flags)
    acc = ((scores[flags==1]>threshold).sum() + (scores[flags==-1]<threshold).sum()) / len(scores)
    return acc.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs_path', default='./data/pairs.txt',
            help='path to pairs.txt')
    parser.add_argument('--dataset', default='./data/processed/lfw',
            help='path to test dataset (LFW)')
    parser.add_argument('--model_path', required=True,
            help='path to trained model weight')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch_size', type=int, default=1024)
    args = parser.parse_args()

    pairs = parse_list(args.pairs_path, args.dataset)

    device = torch.device(args.device)
    model_name = os.path.basename(os.path.normpath(args.model_path)).split('.')[0]
    model_name, emb_dim = model_name.split('_')
    model = ResNetFace(model_name, int(emb_dim)).to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    folds = [p['fold'] for p in pairs]
    flags = [p['flag'] for p in pairs]
    with torch.no_grad():
        print('Extracting features...')
        feat_ls = extract_deep_feature([p['file_l'] for p in pairs], model, device, args.batch_size)
        feat_rs = extract_deep_feature([p['file_r'] for p in pairs], model, device, args.batch_size)
        print('Done!')
        cosine = F.linear(F.normalize(feat_ls), F.normalize(feat_rs))
        scores = torch.diagonal(cosine)

        accuracy = []
        for k in range(1, 11):  # k-fold (k=10)
            val_fold = [i for i, f in enumerate(folds) if f != k]
            test_fold = [i for i, f in enumerate(folds) if f == k]
            val_flags = [flags[i] for i in val_fold]
            test_flags = [flags[i] for i in test_fold]

            threshold = get_threshold(scores[val_fold], val_flags, 10000)
            acc = get_accuracy(scores[test_fold], test_flags, threshold)
            accuracy.append(acc)
            print(f'{k}\t{acc*100.}')
        print(f'Avg. {sum(accuracy)/len(accuracy)*100.}')
