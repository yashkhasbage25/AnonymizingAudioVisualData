---
layout: post
title: Results of FSGAN
---

This post, shows the results of FSGAN.

## FSGAN

Link: https://github.com/YuvalNirkin/fsgan

Paper: https://arxiv.org/pdf/1908.05932.pdf

Video: https://arxiv.org/pdf/1908.05932.pdf

License: CC0-1.0

Paper: 
    ---
    @inproceedings{nirkin2019fsgan,
    title={{FSGAN}: Subject agnostic face swapping and reenactment},
    author={Nirkin, Yuval and Keller, Yosi and Hassner, Tal},
    booktitle={Proceedings of the IEEE International Conference on Computer Vision},
    pages={7184--7193},
    year={2019}
    }
    ---

Colab: https://github.com/YuvalNirkin/fsgan/blob/master/inference/face_swapping.ipynb

Dependencies: git clone https://github.com/YuvalNirkin/face_detection_dsfd

FSGAN replaces the faces on original video with a target face while preserving the lip movement, face movement, eye movement, body gestures, etc. This can be directly applied to full-body videos. 

## Results

Results: https://drive.google.com/drive/folders/1AJlvaqIIHsFZx1kzVb7SRM87NHGPAx9o?usp=sharing

File organization: 
Original Videos: lab_1.mp4, lab_2.mp4 (the demo videos)
Target Faces (Imaginary AI-generated faces): man1.jpg, woman1.jpg
Anonymized videos have the form <target_face>_<original_video>.mp4
For example, man1_lab1.mp4 is formed using man1.jpg and original video lab1.mp4
List of anonymized videos:
1. man1_lab1.mp4 = man1.jpg + lab_1.mp4
2. man1_lab2.mp4 = man1.jpg + lab_2.mp4
3. woman1_lab1.mp4 = woman1.jpg + lab_1.mp4
4. woman1_lab2.mp4 = woman1.jpg + lab_2.mp4

*I believe, this is a "perfect" solution.*

Speed: approx 4 frames per sec, when original video is 1080p and target face is 256x256. 

I was parallely trying Fewshot Face translation GAN, (https://github.com/shaoanlu/fewshot-face-translation-GAN), but it is slow and showed little unnatural output.

Another code method was Articulated-Animation (https://github.com/snap-research/articulated-animation), but the code had some unexpected errors.

A popular method named FaceSwap (https://github.com/deepfakes/faceswap), which is said to be working well by people, needs to be trained. No direct inference models are provided for it. 

DeepFaceLab (https://github.com/iperov/DeepFaceLab) is also popular, and is known to work well. However, it does not have straight forward tutorial/explaination. Also, there is a chance that it needs a manual intervention.

## More workabout links

* The target face has to be converted to a static video. A image can also be provided as input to code, but the output is found to be unstable. In case of a static video, the output is perfect.

* Run instructions: https://github.com/YuvalNirkin/fsgan/wiki/Face-Swapping-Inference

* Creating video from single image: https://stackoverflow.com/questions/25891342/creating-a-video-from-a-single-image-for-a-specific-duration-in-ffmpeg

* Demo colab: https://github.com/YuvalNirkin/fsgan/blob/master/inference/face_swapping.ipynb

* Working with MP4 videos: https://github.com/YuvalNirkin/fsgan/issues/70


## Next Steps

I think we have a proper level of anonymization on both audio and video side. If everyone, agrees that we have reached a sufficient level, I can move it to the pipeline and incorporate into the singularity container. 

Once the setup is ready and well-tested, we can move to modifying our methods.
