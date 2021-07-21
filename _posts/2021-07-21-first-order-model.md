---
layout: post
title: Results from First-Order-Model (FOM)

fompath: assets/img/fom
---

Results from first-order-model (FOM). 

### Introduction

Github: https://github.com/AliaksandrSiarohin/first-order-model

Paper: https://arxiv.org/abs/2104.11280

License: Copyright Snap Inc. 


I chose this repository because it was the only popular repostitory in PyTorch. It was tested by few blogs also. 

There are three entities involved:
1. A source image
2. A driving video
3. A result video

A driving video is the one that needs to be anonymized. A source image is the target person whose face will be used for anonymization. A result video involves the face if target person in the driving video. 

To certain extent the face movements of the speaker are well-captured by FOM. 

The best way to get Source images is https://generated.photos/faces
This website has several AI-generated faces for free. I have used them here. 

### Results

Link to results: https://drive.google.com/drive/folders/19P5HsncUv8_aFO9NrWdwmbXQoTSJoqkP?usp=sharing

#### How are results organized?

If you see the folder: https://drive.google.com/drive/folders/19P5HsncUv8_aFO9NrWdwmbXQoTSJoqkP?usp=sharing

1. The original videos (driving videos) are placed in directory *original_videos*. There are two such videos. These were formed by selecting the facial part from the demo videos. 
    - lab_1_face.mp4
    - lab_2_face.mp4
2. The target images (source images) are placed in directory *target_images*. There are 3 target images of imaginary faces. 
    - man1.jpg
    - woman1.jpg
    - woman2.jpg
3. The result videos (result videos) are placed in *result_videos*. 2 original videos and 3 target images are paired with each other to form 2x3=6 result videos. For example, woman2.jpg is used with lab_1_face.mp4 to get woman2_lab1.mp4 . In similar way, we have named the other videos. 
    - man1_lab1.mp4 
    - man1_lab2.mp4
    - woman1_lab1.mp4
    - woman1_lab2.mp4
    - woman2_lab1.mp4
    - woman2_lab2.mp4

The videos/images are also collectively available here: https://drive.google.com/drive/folders/19P5HsncUv8_aFO9NrWdwmbXQoTSJoqkP?usp=sharing

The original and result videos are totally in-sync. There is no way that their speeds are different. 

### My thoughts

Some face expressions are not captured well. This method only works for face oriented videos. We need to find other method that can work for full body. Overall, this method is just okay, and not very good. Please share your opinions. 

### Next step

Full body anonymization: https://github.com/snap-research/articulated-animation