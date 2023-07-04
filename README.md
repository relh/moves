# MOVES: Manipulated Objects in Video Enable Segmentation

I will publish the cleaned-up code soon. For now here is a brief overview.

## Collect a folder of videos

All you need is a folder of videos to get started!

## Run optical flow bi-directionally between frames offset ~0.5s

I have code that will `ffprobe` a video and spot the framerate, so don't worry about this.

## (Optional) Use off-the-shelf person segmentation to generate people masks.

If your videos don't have people in them, that's fine. Then you can skip this.

## Training

During training we use the above outputs, creating pseudolabels on the fly. We then learn grouping and association from these labels with a simple MLP.

## Citation

If you find our method or this repo helpful, please consider citing our conference paper:

$$
@inproceedings{higgins2023moves,
  title={MOVES: Manipulated Objects in Video Enable Segmentation},
  author={Higgins, Richard EL and Fouhey, David F},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6334--6343},
  year={2023}
}
$$
