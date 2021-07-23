---
layout: post
title: Issues with First-Order-Model (FOM)
---

Some imperfections were detected with FOM. In this post, I list them down, 
their possible reasons/solutions and next steps.

## Issues

* The result from lab_1 was having unnatural lip movement. This may be due to the *cap*. *Need to investigate*

* The result woman1_lab2 has unnatural lip movement. This is because the target image shows teeth to more extent. I observed that if the smile in the target image is big, then it the difficult to remove the smile in the resulting video. Selection of target image is crucial, it should have a smile but not as big as in woman1. Similarly, I also observed that, if the target image has no smile, then the resulting video has a closed mouth throughout. I am sure a normal smile will not cause such defects.

* FOM does not work on full-body. You need to extract the face and apply FOM to it. If we, replace the anonymized face back into original video, it will be unnatural because of the difference in clothing, background, lighting etc. 

## Next Steps

* Test over multiple males and head-wear videos

* Explore full-body anonymizer: https://github.com/snap-research/articulated-animation

* Check other DeepFake Frameworks: FaceSwap and DeepFaceLab