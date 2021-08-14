---
layout: post
title: Advanced settings
---

In this post, I will detail into the advanced settings or the deeper manipulations that are not presented to a general user. 

## Audio anonymizer

1. The audio.py folder provides as large as 40+ parameters, out of which only pitch is present to general user. 

2. The audio codec of the intermediate audio files can be changed in audio.py. Default is wav. 

## Video Anonymizer

1. Swapper: There are several simple and complex options for running fsgan/inference/swap.py (which is the main file that the code runs). The details of several other variations are mentioned in https://github.com/YuvalNirkin/fsgan/wiki/Face-Swapping-Inference . To use this, make changes to rha.py

2. Hider: You can hide the face with other shapes: oval and circle. See arguments of hide_face_robust.py and accordingly change the rha.py