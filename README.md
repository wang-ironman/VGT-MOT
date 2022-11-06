# VGT-MOT

## Title

VGT-MOT ：Visibility guide tracking for online multiple object tracking

## Abstract

Multi-object tracking is the foundation of computer vision and is widely used in video surveillance and autonomous driving scenarios. Most of the existing multi-object tracking methods use Kalman filter to predict the position of the object in the next frame, but there are large errors in the prediction of Kalman filter in the video captured by the moving camera. In addition, how to effectively deal with the occlusion problem in the tracking process also needs further research. To address these problems, a novel joint detection and tracking network, VGT-MOT, is proposed in this paper. To cope with the difficulty of object position prediction due to camera motion, VGT-MOT uses an adjacent-frame object location prediction network to predict the object position in the next frame instead of the traditional Kalman filter. To address the occlusion problem, VGT-MOT guides tracking based on visibility prediction in terms of both similarity metrics and trajectory feature updates. In addition, the predicted visibility is used in the loss calculation of the appearance (Re-ID) branch to reduce the impact of the appearance branch on the detection branch. We evaluated our method on MOT16, MOT17, and MOT20 datasets, and with a single dataset training, MOTA and IDF1 reached 69.7 and 69.4 on MOT17 dataset, and our method has a more advanced performance.

## Experiment results

| Dataset | MOTA | IDF1 | IDS  |  MT  |  ML  |
| :-----: | :--: | :--: | :--: | :--: | :--: |
|  MOT16  | 71.0 | 70.3 | 954  | 305  | 157  |
|  MOT17  | 69.7 | 69.4 | 2841 | 988  | 476  |
|  MOT20  | 60.9 | 63.4 | 2777 | 711  | 172  |

All of the results are obtained on the [MOT challenge](https://motchallenge.net/) evaluation server under the “private detector” protocol. Our method has a more advanced performance.

## Installation

- Clone this repo, and we'll call the directory that you cloned as 

- Install dependencies. We use python 3.6 and pytorch >= 1.1.0

```
conda create -n VGTMOT
conda activate VGTMOT
conda install pytorch==1.1.0 torchvision==0.2.2 cudatoolkit=9.0 -c pytorch
cd ${ROOT}
pip install cython
pip install -r requirements.txt
```

- We use DCNv2, correlation, channelNorm  in our backbone network. The Path is ${ROOT}/models/networks.
- We also use apex for FP16 training.

```
cd ${ROOT}/models/networks/DCNv2_new
./make.sh
cd ${ROOT}/models/networks/correlation_package
python setup.py install
cd ${ROOT}/models/networks/channelnorm_package
python setup.py install
cd ${ROOT}/apex
python setup.py install
```

## Data preparation

MOT17 dataset can be downloaded at [MOTChallenge](https://motchallenge.net/data/MOT17/).

We uses two CSV files to organize the MOT17 dataset: one file containing annotations and one file containing a class name to ID mapping. 

We provide the two CSV files for MOT17 with codes in the CTRACKER_ROOT/data, you should copy them to MOT17_ROOT before starting training. 

```
MOT17_ROOT/
        |->train/
        |    |->MOT17-02/
        |    |->MOT17-04/
        |    |->...
        |->test/
        |    |->MOT17-01/
        |    |->MOT17-03/
        |    |->...
        |->train_annots.csv
        |->train_labels.csv
```

MOT17_ROOT is your path of the MOT17 Dataset.

## Annotations format

The CSV file with annotations should contain one annotation per line.
Images with multiple bounding boxes should use one row per bounding box.
Note that indexing for pixel values starts at 0.
The expected format of each line is:

```
path/to/image.jpg,id,x1,y1,x2,y2,vis,class_name
```

## Class mapping format

The class name to ID mapping file should contain one mapping per line.
Each line should use the following format:

```
class_name,id
```

Indexing for classes starts at 0.
Do not include a background class as it is implicit.

For example:

```
person,0
dog,1
```

## Training

The network can be trained using the `train.py` script. For training on MOT17, use

```
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --master_port 25469  train.py --batchsize 2        --root_path {data_root} --csv_train train_annots.csv
```

By default, testing will start immediately after training finished.

## Testing and validation

Run the following commands to start testing:

```
python test.py
python test_half.py
```

## Pretrained models and baseline model

- **Pretrained models**

  DLA-34 COCO pretrained model: [DLA-34 official](https://drive.google.com/file/d/1pl_-ael8wERdUREEnaIfqOV_VF2bEVRT/view). 

  ```
  ${ROOT}
     └——————models
             └——————ctdet_coco_dla_2x.pth
  ```

  

- **Baseline model**

  ```
  ${FAIRMOT_ROOT}
     └——————models
             └——————model_final.pth
  ```
