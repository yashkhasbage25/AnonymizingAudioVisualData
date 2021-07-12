---
layout: post
title: Robust Face Hider and the choices for face detectors
---

We propose a heuristic to make the face-hider more robust and finally, summarize the face detectors we chose.

### Algorithm for robust detection

Let the set of boxes at time $$t$$ be $$b_t$$. With the notion of time gap $$\delta t$$
, the past and future frames are $$b_{t - \delta t}$$ and $$b_{t + \delta t}$$, respectively. 


The idea is simple and straigth:
See the future and past frames, if there are boxes which are close, see if there exists such a box near to them in current frame. If there is no such box, create one. 


```python
for b1 in prev_boxes:
    for b2 in next_boxes:
        if distance(b1, b2) < threshold:
            avg_box = average_box(b1, b2)
            exists = False
            for b3 in this_boxes:
                if distance(avg_box, b3) < threshold:
                    exists = True
                    break
            if not exists:
                this_boxes.append(b3)
```


It may not be an all-situation-single-solution, idea. But definitely, it is sufficient for lab videos. 


### Face detector summary

1. First, we used a Haar-Cascade face detector. It is openly available in the popular library OpenCV. 

In our demo videos, *it mispredicted few frames of news_2.mp4*. Specifically, it predicted guitar head as face. 


Its accuracy can be considered as *99% on lab videos*. 

Link: https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html

2. Second, we shifted to MTCNN, a more popular and recent detector. 

It also predicted some other objects as faces (with low confidence), but we decided to hide every such prediction.

Its accuracy can be considered as *100% on lab videos*. But fails usually (with 5% chance. It happens when faces are occluded or tilted at extreme angles) for Ellen DeGeneres Show. 

Links: 

GitHub: https://github.com/ipazc/mtcnn (No license mentioned here, instead mentioned in PyPi page)

Anaconda: https://anaconda.org/conda-forge/mtcnn (license: MIT, I used directly from here)

Cite: Zhang, K., Zhang, Z., Li, Z., and Qiao, Y. (2016). Joint face detection and alignment using multitask cascaded convolutional networks. IEEE Signal Processing Letters, 23(10):1499–1503.


##### If method 2 worked 100% on lab videos, why did we think further? 

* Because, Neural Network Face Detectors are seen to fail at extreme tilt angles and occluded faces. This was observed on the DeGeneres Show also. We should avoid any chance of such mistake. 

3. Third, we implement a heuristic based algorithm that *ensures persistency* in hiding faces. With this we were successful in anoymizing a random clip from DeGeneres. 
Overall accuracy on DeGeneres: not known.

| Method | Lab Videos | News Videos | Ellen DeGeneres |
|:------:|:----------:|:-----------:|:---------------:|
| Haar Cascade Detector | :heavy_check_mark: | :x: | :x: |
| MTCNN | :heavy_check_mark: | :heavy_check_mark: | :x: |
| Robust (ours) | :heavy_check_mark: | :heavy_check_mark: | Not evaluated seriously, but looks good |