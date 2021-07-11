---
layout: post
title: Robust Face Hider
---

We propose a heuristic to make the face-hider more robust

### Algorithm

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