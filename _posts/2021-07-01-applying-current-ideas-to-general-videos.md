---
layout: post
title: Applying Current Ideas to General Videos
---

AttGAN is definitely a good tool for RedHen Transformer. Vid2vid also showed a good example of how a person's clothes can be changed. 
What are difficulties of applying them directly on the general lab-environment videos?

As a first step, we willl anonymize 6 videos:
* 3 are of students describing some event
* rest 3 are more of a TV show

#### Difficulties and Solutions

* The videos are high resolution. Face occupies only a small part of it. Using AttGAN over complete video will make it slow and memory-consuming. 
Thus, we need to extract the face first, transform the face, and place it back at right location in video. There are several recent high-accuracy 
face detectors, however with simple lab-based videos, we don't need any sophisticated method. Haar-Cascade detector (https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html) will work in such cases. 

* AttGAN requires the original attributes of the person to be anonymized. This was not an issue when we tried it on CelebA dataset, as the dataset already provides 
such attribution. However, we need to get attribution for general videos. This can be accomplished by creating a classifier that predicts the attributes. 
How to train such classifier? We can use the CelebA dataset, because it has such attribution. We train the classifier on CelebA dataset while targetting the desired attributes. 

* Input size: the input size for AttGAN is _strictly_ 112x112. However this need not be the case when we extract face from videos. Resizing of an image can definitely change dimensions
but the ratio of height-to-width has to remain constant for maintaining its naturality. Hence, the face-images have to be padded (most probably, with zeros) to form square, and then resized to 112x112. Also, the zero padding has to be removed 

* vid2vid has to be applied to video after face anonymization. 