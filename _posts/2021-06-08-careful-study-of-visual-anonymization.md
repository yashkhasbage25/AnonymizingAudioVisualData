---
layout: post
title: Careful Study of Visual Anonymization Papers
---

## Intention

This post is for carefull distinguishing between what is readily available, what needs to be ported to PyTorch, what needs to be implemented from scratch, and what is unexplored.

There are 4 papers that can fulfill our task: 

| Title | Link | Code Link | Framework | 
|-------|------|-----------|-----------|
| DeepPrivacy: A Generative Adversarial Network for Face Anonymization | https://arxiv.org/abs/1909.04538 | https://github.com/hukkelas/DeepPrivacy | PyTorch |
| AttGAN: Facial Attribute Editing by Only Changing What You Want | https://arxiv.org/abs/1711.10678 | https://github.com/LynnHo/AttGAN-Tensorflow | TensorFlow |
| StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation | https://arxiv.org/abs/1711.09020 | https://github.com/cosmic119/StarGAN | PyTorch |
| StarGAN v2: Diverse Image Synthesis for Multiple Domains | https://arxiv.org/abs/1912.01865 | https://github.com/clovaai/stargan-v2 | PyTorch |


#### I will summarize the papers and also note down what can be learned from these papers.

What we need to adopt from these papers: metrics, datasets

## DeepPrivacy: A Generative Adversarial Network for Face Anonymization

* Generator is U-Net
* High resolution is possible only with Progressive GAN training
* requires bounding box annotation of privacy sensitive area, sparse pose estimation of the face, containing keypoints for the ears, eyers, nose, shoulder
* authors provide a new dataset: Flickr Diverse Faces (FDF) which satisfy their requirements, www.github.com/hukkelas/FDF
* they test over WIDER-Face dataset, http://shuoyang1213.me/WIDERFACE/
* metric: Average Precision (AP)
* compare with other methods: 8x8 pixelation, heavy blur, black-out
* Generator: U-net, same as Progressive GAN
* Discriminator: same as Progressive GAN
  * background information as conditional input to the start of discriminator, making the input have six channels instead of three
  * include pose information at each resolution of the discriminator
  * remove the mini-batch standard deviation layer

Pros:
* models readily available
* works on general datasets

Cons:
* Difficult training
* No control over anonynmization

## AttGAN: Facial Attribute Editing by Only Changing What You Want

* Why not use Fader networks? In Fader Networks, an adversarial process is introduced to force the latent representation to be invariant to the attributes. However, the attributes portray the characteristics of a face image, which implies the relation between the attributes and the face latent representation is highly complex and closely dependent. Therefore, simply imposing the attribute-independent constraint on the latent representation not only restricts its representation ability but also may result in information loss, which is harmful to the attribute editing.

#### Testing Formulation

$$x^a$$ is tha face image with $$n$$ binary attributes $$a = [a_1, ..., a_n]$$. 

$$z = G_{enc} (x^a)$$

$$b = [b_1, ..., b_n]$$ are another attributes to be achieved

$$x^{\hat{b}} = G_{dec}(z, b)$$

#### Training Formulation

An _attribute classifier_ is used to constrain the generated image $$x^{\hat{b}}$$ to correctly own the desired attributes. Meanwhile, the _adversarial learning_ is employed on $$x^{\hat{b}}$$ to ensure its visual reality. 

On the other hand, an eligible attribute editing should only
change those desired attributes, while keeping the other details
unchanged. To this end, the reconstruction learning is introduced 
to 1) make the latent representation z conserve enough 
information for the later recovery of the attribute-excluding
details, 2) enable the decoder $$G_{dec}$$ to restore the attribute 
excluding details from z.

$$ x^{\hat{a}} = G_{dec} (z, a) $$

#### Extension for Attribute Style Manipulation

_Converting binary attributes to continuous_

Style controllers: $$ \theta = [\theta_1, \theta_2, ..., \theta_n] $$. 
We will bind each $$\theta_i$$ and the $$i$$th attribute, and maximize the mutual information between the controllers and the output images to make them highly correlated. We add style controllers and a style predictor Q, and the attribute editing is reformulated as $$ x^{\hat{\theta}\hat{b}} = G_{dec}(G_{enc}(x^a, \theta, b))$$

#### Experiments

Dataset: CelebA

13 Attributes: Bald, Bangs, Black, Hair, Blond Hair, Brown Hair, Busy Eyebrows, Eye-glasses, Gender, Mouth Open, Mustache, No beard, Pale Skin, Age 