---
layout: post
title: Results of DeepPrivacy
---

We use DeepPrivacy to anonymize videos. 

### DeepPrivacy 

Link: https://github.com/hukkelas/DeepPrivacy
Citation: 

    ---
        @InProceedings{10.1007/978-3-030-33720-9_44,
        author="Hukkel{\aa}s, H{\aa}kon
        and Mester, Rudolf
        and Lindseth, Frank",
        title="DeepPrivacy: A Generative Adversarial Network for Face Anonymization",
        booktitle="Advances in Visual Computing",
        year="2019",
        publisher="Springer International Publishing",
        pages="565--578",
        isbn="978-3-030-33720-9"
        }
    ---

License: MIT

The faces are removed from original videos, and a generator is used to fill the blank part. 

This ensures that the anonymized faces cannot be recovered. But, the new face is not restricted to a particular person. Hence, with time, the face in video changes in considerable quantity.

Also, the facial expression are totally lost. Hence, *DeepPrivacy is almost like Hiding-faces*.

Demo: https://drive.google.com/drive/folders/1obOc1rA0ydOH4zAI6BmbKy687s3oz41O?usp=sharing

### IRB demos

Hiding face: https://drive.google.com/drive/folders/1Mrt--AqTC9ohYD457peYPZw0RvBHYSOK?usp=sharing

DeepPrivacy: https://drive.google.com/drive/folders/1obOc1rA0ydOH4zAI6BmbKy687s3oz41O?usp=sharing

### What next?

Check several other DeepFake codes. Most of them are in tensorflow. But, this time, let's directly test them on demo videos. 