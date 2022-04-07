# pytorch-uniformface
A PyTorch implementation of [UniformFace(CVPR2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Duan_UniformFace_Learning_Deep_Equidistributed_Representation_for_Face_Recognition_CVPR_2019_paper.pdf)

**This is not author's official repository**

## preprocess
Pretrained MTCNN model used in this repository ([facenet-pytorch](https://github.com/timesler/facenet-pytorch)) is not same as the model used in original sphereface.
So the test result is not comparable with other repositories:
- https://github.com/wy1iu/sphereface
- https://github.com/clcarwin/sphereface_pytorch
 
Download CASIA-WebFace and LFW dataset to ./data directory and run
```
./docker_build.sh && ./docker_run.sh
```
in the docker container, run
```
python ./code/preprocess.py --dataset ./data/CASIA-WebFace --output_dir ./data/processed/CASIA-WebFace
python ./code/preprocess.py --dataset ./data/lfw --output_dir ./data/processed/lfw
```
The preprocessing took about 3 hours for CASIA-WebFace in the following environment
- RTX2080Ti
- Intel Core i9-9900K

(maybe there is more efficient preprocessing approach e.g. batch face detect)

## train
In the docker container, run
```
python ./code/train.py --config ./code/config.yaml --logdir ./log
```

alpha is weight of uniform loss, beta is update ratio of class centers
- blue: alpha=0.0, beta=0.0 (normal sphereface)
- orange: alpha=1.0, beta=0.001 (with uniform loss and moving class centers)
- red: alpha=1.0, beta=0.0 (with uniform loss)

when beta is 0.1 or 0.01, distances between classes become very large and uniform loss becomes nan

### accuracy 
![train_accuracy](https://user-images.githubusercontent.com/46083154/162228196-578b57b7-8bb8-425b-9952-37f3b6fa3bbb.svg)
### cross entropy loss
![train_ce_loss](https://user-images.githubusercontent.com/46083154/162228298-85490232-4a39-4929-a984-9770e68716fa.svg)
### uniform loss
![train_uni_loss](https://user-images.githubusercontent.com/46083154/162228359-52cb00b1-5899-47d1-8156-d88526061c55.svg)

## test
Download http://vis-www.cs.umass.edu/lfw/pairs.txt to ./data

In the docker container, run
```
python ./code/test.py --model_path ./log/seedhogehoge_hogefuga/{model_name}_{emb_dim}.pt
```
(directory name is generated by config.yaml using ./code/train.py:32)

### result(10-fold)
- LFW accuracy

|                    |   Avg. |    1   |    2   |   3    |   4    |   5    |   6    |   7    |   8    |   9    |   10   |
| ------------------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| alpha=0.0/beta=0.0 | 98.967 | 99.000 | 99.000 | 99.000 | 98.500 | 99.000 | 99.000 | 99.000 | 98.667 | 99.500 | 99.000 |
|alpha=1.0/beta=0.001| 98.733 | 98.667 | 98.333 | 98.667 | 98.000 | 98.833 | 98.667 | 98.667 | 98.833 | 99.500 | 99.167 |
| alpha=1.0/beta=0.0 | 98.683 | 98.333 | 98.333 | 98.667 | 98.667 | 98.500 | 98.500 | 98.667 | 98.833 | 99.666 | 98.667 |

- average of minimum class distance for all class centers

|                    |   Min Avg. |
| ------------------ | ---------  |
| alpha=0.0/beta=0.0 |   0.1516   |
|alpha=1.0/beta=0.001|   14.1534  |
| alpha=1.0/beta=0.0 |   0.1638   |


## ToDo
The test result is not good, so I will explore other hyperparameters.
- increase batch size (make more stability of moving class centers)
- increase training step size (make convergence and higher accuracy)
