---
layout: post
title: The output till Eval-1 and meeting summary for 7 July
---

This post summarizes, the pre-eval-1 meeting and the results at the stage of Eval-1.

Some sections of this post will match with the previous posts as this post also contains summary of them. 


### Audio Anonymization

We use the sox module for python. moviepy module was used for handling video and audio. There are a large variety of effects offered by sox, and each of them has some parameters. 

As a general observation, the pitch was found to have a dominant role in anonymizing. Pitch itself was found to be sufficient. 

@Karan is going to point some method form the VoicePrivacy Challenge. 


### Face Anoymization

The AttGAN did not work well on our videos. Hence, we had to completely reject AttGAN. (The code for AttGAN is ready, if it will be useful anytime in future)

In the earlier post, we tried hiding faces with Haar-Cascade detector. Now, we tried MTCNN (https://github.com/ipazc/mtcnn, https://arxiv.org/abs/1604.02878), which is much reliable detector. We detect the faces and hide them with rectangle/oval/circle. The face detector is inherently good even for multi-face images. However, we need to test its robustness. 

Later, we also tried Facelet-Bank (https://github.com/dvlab-research/Facelet_Bank). This method was able to make some succesful face editing, but the editing was not found to anonymize enough. The faces were properly recognizable even after feature editing. 


### Next Steps

1. First make `hiding-faces` reliable. Test over various videos. 

2. Implement the method to be suggested by @Karan. 


# Meeting summary

1. Is the face hidden by rectangles recoverable? Is there some layering of rectangle over actual video, which can be removed to get the original video back?

No. I have manually *replaced* the pixels and they are totally non-recoverable. 

2. The `hiding-faces` has to be a 99.999% reliable method. We cannot miss even a single frame. 

Although MTCNN is used for prediction, there are always chances that it may not work under occlusions or off-camera angles. We need to test under wide variety of videos. 

MTCNN is observed to sometimes detect some objects as faces (although with low confidence). As suggested by Prof. Mark, we will hide all the predictions even if they are false. Also, the false prediction was that of a guiat head being detected as face, which is something very rare in lab settings. 

Another heuristic is to track the motion of rectangles. When the predictions change rapidly for a few frames, it can also mean something wrong. 

MTCNN inherently, is capable of detecting multiple faces in a single image, and is also robust to face angles. However we need to still verify its robustness. We are going to test it either over public domain videos or data to be shared by @Daniel. 

3. Face editing with simple features does not seem to anonymize well. Hence, we need to shift to replacing faces. 

Yes, there are works on DeepFake, with two directions: either you can replace with a targetted person or with an imaginary face. 

The work by https://github.com/hukkelas/DeepPrivacy is a readily available code for imaginary face replacement. We can test it in short time and see its results. 

Regarding DeepFake, I need to search for available methods (I am personally excited to explored DeepFakes). As commented by @Prof Mark, we need to replace with an ordinary face instead of a celebrity. 



