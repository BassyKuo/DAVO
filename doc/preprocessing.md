## Prepare Traing/Testing Data

There are three types of inputs used in DAVO:

1. [RGB frames](#rgb-frames)
2. [Optical flows](#optical-flows)
3. [Semantic segmentations](#semantic-segmentations)

Let's check how to prepare them for DAVO.

## RGB frames

#### KITTI Odometry
For [KITTI](http://www.cvlibs.net/datasets/kitti), you need to download the [KITTI odometry dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) first, 
and unzip them into the `$kitti_raw_odom` folder (for example, `./kitti_odom/`). Then run the following command:

```bash
# Set folder paths.
export kitti_raw_odom="./kitti_odom/"         # make sure kitti_odom/ includes `sequences/' and `poses/'
export kitti_odom_dump="./kitti_odom_dump/"

# Dump the KITTI VO dataset.
python data/preprocess.py --dataset_dir=$kitti_raw_odom --dataset_name='kitti_odom' --dump_root=$kitti_odom_dump --seq_length=3 --img_width=416 --img_height=128 --num_threads=8
```
###### [REMARK]: Add argument `--generate_test` in your command to genrate testing data.

#### (optional) Cityscapes Sequence

For [Cityscapes](https://www.cityscapes-dataset.com/), download the following packages: (1) `leftImg8bit_sequence_trainvaltest.zip` (324GB), (2) `camera_trainvaltest.zip` (1.9MB), and unzip them into the `$cityscapes_raw` folder.
Then run the following command:

```bash
# for Cityscapes dataset
python data/preprocess.py --dataset_dir=$cityscapes_raw --dataset_name='cityscapes' --dump_root=$cityscapes_odom_dump --seq_length=3 --img_width=416 --img_height=128 --num_threads=8
```

Note that for Cityscapes the img_height is set to 171 because we crop out the bottom part of the image that contains the car logo, and the resulting image will have height 128.

###### [REMARK]: Add argument `--generate_test` in your command to genrate testing data.


## Optical flows

In DAVO, we use [FlowNet2.0](https://www.zpascal.net/cvpr2017/Ilg_FlowNet_2.0_Evolution_CVPR_2017_paper.pdf) (CVPR'2017) to generate optical flows for each input frames-pair.
Resources:

* Pytorch: https://github.com/NVIDIA/flownet2-pytorch
* Tensorflow: https://github.com/sampepose/flownet2-tf (paper used)

There are two (or four) optical flows should be generated into `$kitti_odom_dump/$seq_name/<6-digital-id>-flownet2.npy`:

1. target -> source0 (i.e. the optical flow from `000001` to `000000`)
2. target -> source1 (i.e. the optical flow from `000001` to `000002`)
3. source0 -> target (i.e. the optical flow from `000000` to `000001`) [optional]
4. source1 -> target (i.e. the optical flow from `000002` to `000001`) [optional]

#### Example code:

```python
# ... (loading $kitti_odom_dump/$seq_name/000001.png) ...
im = scipy.misc.imread( "$kitti_odom_dump/$seq_name/000001.png" )
h,w,c = im.shape
src0, tgt, src1 = im[:,:w//3,:], im[:,w//3:w//3*2,:], im[:,w//3*2:,:]

# ... (generate flows) ...
src0_tgt = flownet2.predict(input_a=src0, input_b=tgt)
src1_tgt = flownet2.predict(input_a=src1, input_b=tgt)
tgt_src0 = flownet2.predict(input_a=tgt, input_b=src0)
tgt_src1 = flownet2.predict(input_a=tgt, input_b=src1)

# ... (save in npy file) ...
flows = [
  src0_tgt,     # dtype=np.float32. shape=(height,width,2)
  src1_tgt,     # dtype=np.float32. shape=(height,width,2)
  tgt_src0,     # dtype=np.float32. shape=(height,width,2)
  tgt_src1,     # dtype=np.float32. shape=(height,width,2)
  ]

all_flows = np.stack(flows)       # shape=(4,height,width,2)
np.save("$kitti_odom_dump/$seq_name/000001-flownet2.npy", all_flows)
```

## Semantic segmentations

In DAVO, we use [DeepLab3+](https://github.com/tensorflow/models/tree/master/research/deeplab) (ECCV'2018) to generate semantic segmentations for each frame.
There are three segmentation label maps should be generated into `$kitti_odom_dump/$seq_name/<6-digital-id>-seglabel.npy`:

1. source0 segmentation labels (i.e. the label map of `00000`)
2. target segmentation labels (i.e. the label map of `00001`)
3. source1 segmentation labels (i.e. the label map of `00002`)

#### Example code:

```python
# ... (loading $kitti_odom_dump/$seq_name/000001.png) ...
im = scipy.misc.imread( "$kitti_odom_dump/$seq_name/000001.png" )
h,w,c = im.shape
src0, tgt, src1 = im[:,:w//3,:], im[:,w//3:w//3*2,:], im[:,w//3*2:,:] 

# ... (generate segmentations) ...
src0 = deeplab.predict(src0)
tgt  = deeplab.predict(tgt)
src1 = deeplab.predict(src1)

# ... (save in npy file) ...
seglabels = [
  src0,     # dtype=np.float32. shape=(height,width,1)
  tgt,      # dtype=np.float32. shape=(height,width,1)
  src1,     # dtype=np.float32. shape=(height,width,1)
  ]

all_seglabels = np.stack(seglabels)       # shape=(3,height,width,1)
np.save("$kitti_odom_dump/$seq_name/000001-seglabel.npy", all_seglabels)
```
