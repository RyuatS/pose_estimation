# pose_estimation
## part1. Introduction
---
Implementation of human pose estimator for COCO dataset in Tensorflow.<br>
COCOデータセットの人間の姿勢推定のTensorflowでの実装
- [x] You can define the model yourself
- [x] Download Backbone network
- [x] COCO dataset converter to tfrecord
- [x] Training Pipeling
- [ ] Visualize script (Future Implementation)

## part2. Train on COCO dataset.
---
以下に、訓練するためのステップを示します。
1. まず、レポジトリをクローンする
```
$git clone https://github.com/RyuatS/pose_estimation.git
$cd pose_estimation
```

2. 以下のコードを実行して、使いたいbackboneネットワークの重み(default: resnet_v1_50)とCOCOデータセット(default: val)をダウンロードする。もし、手動でダウンロードした場合は、重みは、`./backbone_checkpoints/`に入れ、データセットは`./data/cocodevkit/dataset/`に入れる。
```
$cd data
$python setup.py --pre_trained=resnet_v1_50 \
                    --dataset_type=val
```

3. データセットを、tfrecordに変換する。(ディレクトリ`data`内で実行)
```
$python build_coco_data.py --image_dir=./cocodevkit/dataset/val2017 \
                             --annotation_file=./cocodevkit/dataset/annotations/person_keypoints_val2017.json \
                             --output_dir=./cocodevkit/tfrecord
```

4. 学習.
```
$python train.py 
```
