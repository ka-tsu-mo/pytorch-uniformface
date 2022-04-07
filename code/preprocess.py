import argparse
import os
from glob import glob
from tqdm import tqdm

import numpy as np
import torch
import skimage
from skimage.transform import SimilarityTransform, warp
from PIL import Image
from facenet_pytorch import MTCNN


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True,
            help='path to base directory of target dataset')
    parser.add_argument('--output_dir', required=True,
            help='path to output directory of processed data')
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # https://github.com/wy1iu/sphereface/blob/master/preprocess/code/face_detect_demo.m
    # thresholds = [0.6, 0.7, 0.9]
    # factor = 0.85
    # Using facenet-pytorch with above setting, you will failed to detect some faces
    # e.g. lfw/Marilyn_Monroe/Marilyn_Monroe_0001.jpg
    # This causes different test set with other repositories and the test reuslt will not be comparable.
    # To avoid this issue, threshold and factor is default setting of MTCNN
    thresholds = [0.6, 0.7, 0.7]
    factor = 0.709
    mtcnn = MTCNN(
        thresholds=thresholds,
        factor=factor,
        device=torch.device('cuda'))

    coord5point = [[30.2946, 51.6963],
                   [65.5318, 51.5014],
                   [48.0252, 71.7366],
                   [33.5493, 92.3655],
                   [62.7299, 92.2041]]
    coord5point = np.array(coord5point)
    # convert coord5point(112x96) to coord5point(112x112)
    # reference: https://github.com/deepinsight/insightface/blob/205a37e1c11fbe58b721208b8bd79f42fe448a70/src/common/face_preprocess.py
    coord5point[:, 0] += 8.0
    tform = SimilarityTransform()

    for file in tqdm(glob(os.path.join(args.dataset, '*', '*'))):
        image = Image.open(file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        boxes, probs, points = mtcnn.detect(image, landmarks=True)
        box, prob, point = mtcnn.select_boxes(
                boxes, probs, points, image, method='center_weighted_size'
                )
        label = os.path.normpath(file).split(os.sep)[-2]
        if not os.path.exists(os.path.join(args.output_dir, label)):
            os.makedirs(os.path.join(args.output_dir, label))
        if point is not None:
             tform.estimate(point[0], coord5point)
             new_image = warp(np.array(image), tform.inverse, output_shape=(112, 112))
             new_image = Image.fromarray(skimage.img_as_ubyte(new_image))
             base_name = os.path.basename(os.path.normpath(file))
             output_file = os.path.join(os.path.join(args.output_dir, label, base_name))
             new_image.save(output_file)
