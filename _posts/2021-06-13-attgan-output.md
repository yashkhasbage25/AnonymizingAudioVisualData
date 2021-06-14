---
layout: post
title: AttGAN output for CelebA dataset, some examples
---

## Altering Single Feature

One feature changed in each subplot

Features: Bald, Bangs, Black, Hair, Blond Hair, Brown Hair, Busy Eyebrows, Eye-glasses, Gender, Mouth Open, Mustache, No beard, Pale Skin, Age

Explaination: 
First one is original image
Second one is constructing the same image give its original attributes. Ideally, it should 

{% for image in site.static_files %}
    {% if image.path contains 'img/attgan/sample_testing/' %}
![]({{ site.baseurl }}{{ image.path }})
    {% endif %}
{% endfor %}

## Altering Multiple Attributes

Pale Skin and Male feature changed 

{% for image in site.static_files %}
    {% if image.path contains "img/attgan/sample_testing_multi_Pale_Skin__Male" %}
![]({{ site.baseurl }}{{ image.path }})
    {% endif %}
{% endfor %}

## Sliding Attributes

Sliding variation with Male feature

{% for image in site.static_files %}
    {% if image.path contains 'img/attgan/sample_testing_slide_Male' %}
![]({{ site.baseurl }}{{ image.path }})
    {% endif %}
{% endfor %}
