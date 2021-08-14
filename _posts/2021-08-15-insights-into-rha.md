---
layout: post
title: Insights into the internals of rha.py
---

In this post, I will describe the internal processing of rha.py. 

## Visual Anonnymization

Visual anonymization can happen in two ways: Face Hiding and Face Swapping. Both of the iterate over the frames and make changes to individual frames. The intermediate processing data is always stored in folders named "anon_temp_{input_video_name}_{face_name}". These folders get deleted once the processing is completed. However, you need to manually delete them if you stop the processing using keyboard interrupts. 

We begin by getting metadata of video like start times of the streams. 

### Face Hider

While iterating over each frame, the face locations of the current frame are computed and stored.

However, the MTCNN, was found to loose some intermediate frames. This can be easily rectified by seeing the predictions of previous and next frames. If there exists a detection in previous and next frames but not in the current frame, then we need to artificially create the prediction for the current frame. 

However, it is not necessary to always see the immediately next or previous frame. We can see some frames time_delta steps forward or backward. 

Thus, we compare all the frames in `nframe - args.time_delta` and `nframe + args.time_delta` to see verify the frames of `nframe`. If there are face-boxes in the `nframe - args.time_delta` and `nframe + args.time_delta`, which are near and there does not exist a face-box in `nframe` near the center of the two, we need to add such box. The box should have average parameters of the previous and next boxes. 

The frames not having either of future or past frames are adjusted with whatever information is available from future or past. 

Hider is much more robust as compared to the swapper. For example, it can detect faces at extreme angles of tilt. 

### Face Swapper

A detailed understanding of Face Swapper will require study of FSGAN. We won't go into its details. 

FSGAN iterates over each frame and swaps the faces for individual frames. Note that the swapping for the current frame has nothing to do with future or past frames. 

### Extracting Audio

The audio from video are extracted using ffmpeg and saved as .wav files.

### Anonymizing Audio

The wav file is anonymized using Sox transforms.

### Combining Audio and Video

The anonymized audio and video are combined using the ffmpeg while considering the start times of each stream.

