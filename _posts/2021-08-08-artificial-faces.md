---
layout: post
title: Using AI generated faces for FSGAN targets
---

In this post, I will list down some tips for selecting anonymuos faces for anonymization.

### Where to get anonymous faces (target faces)

The website https://generated.photos/faces provides a variety of artificially generated faces. You can signup for the website and there are really a lot of faces.

### What (target) faces are good for FSGAN?

* The faces provided by generated.photos are properly aligned and with plain background, which are ideal for FSGAN. 

* The face should have a slight smile. A no-smile face or big-smile may degrade the quality of output. 

* The skin-color of the target face is used. Hence, our face-bank should have wide ranges of skin-colors. Same for lip-colors, eyebrows and eye-color.

* However, the face hair (beard, moustache or on-head) of target face is not used. Hence, it is fruitless to choose faces with variety of hair-looks. 

* Anything away from the target face or in target's background won't affect at all. Hence, the tag "Image by generated.photos" is not a issue for us. 

* You can also have aged people and kids. 

* The old people in generated.photos are not very old. Hence a near-old face is suggested for old face.

### Special Cases

Special cases arise when there are several artifacts appearing on the face. 
The way to handle them is to have same artifacts on target face too. 

* We observed that it is more suitable to anonymize a face wearing glasses with target face wearing glasses. Similar for glasses with reflection. 

### Our face-bank

Link: https://drive.google.com/drive/folders/1EGiVI3fMLwNiYG-Es-Sy5qqie9Co2eZI?usp=sharing

We have the following categories of faces:
1. black_female
2. black_male
3. white_male
4. white_female
5. old
6. kids
7. glasses

On an average, there are 5 faces in each category. 