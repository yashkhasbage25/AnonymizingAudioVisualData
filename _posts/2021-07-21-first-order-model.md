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

Note that the videos are out of sync only because of the video-player in browser. The videos in gif format can be found here: https://drive.google.com/drive/folders/19P5HsncUv8_aFO9NrWdwmbXQoTSJoqkP?usp=sharing
and they are in-sync. No difference in speed of any video. 

### My thoughts

Some face expressions are not captured well. This method only works for face oriented videos. We need to find other method that can work for full body. Overall, this method is just okay, and not very good. Please share your opinions. 

| Driving Video | Source Image | Result Video |
|:-------------:|:------------:|:------------:|
| ![lab1](/{{ page.fompath }}/lab_1_face.gif "lab1") | ![man1](/{{ page.fompath }}/man1.jpg "man1") | ![man1_lab1](/{{ page.fompath }}/man1_lab1.gif "man1_lab1") |
| ![lab1](/{{ page.fompath }}/lab_1_face.gif "lab1") | ![woman1](/{{ page.fompath }}/woman2.jpg "woman1") | ![woman1_lab1](/{{ page.fompath }}/woman2_lab1.gif "woman2_lab1") |
| ![lab1](/{{ page.fompath }}/lab_1_face.gif "lab1") | ![woman2](/{{ page.fompath }}/woman3.jpg "man1") | ![woman2_lab1](/{{ page.fompath }}/woman3_lab1.gif "woman3_lab1") |
| ![lab2](/{{ page.fompath }}/lab_2_face.gif "lab2") | ![man1](/{{ page.fompath }}/man1.jpg "man1") | ![man1_lab2](/{{ page.fompath }}/man1_lab2.gif "man1_lab2") |
| ![lab2](/{{ page.fompath }}/lab_2_face.gif "lab2") | ![woman1](/{{ page.fompath }}/woman2.jpg "woman1") | ![woman1_lab2](/{{ page.fompath }}/woman2_lab2.gif "woman1_lab2") |
| ![lab2](/{{ page.fompath }}/lab_2_face.gif "lab2") | ![woman2](/{{ page.fompath }}/woman3.jpg "woman2") | ![woman2_lab2](/{{ page.fompath }}/woman3_lab2.gif "woman2_lab2") |
