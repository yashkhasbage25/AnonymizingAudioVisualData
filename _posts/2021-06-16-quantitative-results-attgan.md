---
layout: post
title: Quantitatively Evaluating the Anonymization from AttGAN
---

## Quantitatively Evaluating the Anonymization from AttGAN

#### Metric

DeepPrivacy paper did use a metric "AP", however, their neither properly described it in paper, nor did they provid any code for it. However, AP is understood as Average Precision. We can still make some implementations of it, but it will be still questionable. 

Prof Mark and Karan had already suggested a metric based on distances in latent feature space. We can very well use KNN algorithm to formalize this metric.

CelebA consists of thousands of celebrities. We will train a KNN with features of data. We then anonymize the data and get the features from anonymized images. *The job of anonymizer is that the features of anonymized image for person A, should be pushed far from the features of original images of person A.* In KNN, we will have a matching of features: features of anonymized image of A, and features of all original images in dataset. *The more the mismatch, the better the anonymizer.*

First, we see how each single attribute can affect. Then, we can see the top-3 to 5 attributes which work best. We can form various combinations of these attributes as final set of attributes to be presented. 

Attributes: "Bald",
        "Bangs",
        "Black_Hair",
        "Blond_Hair",
        "Brown_Hair",
        "Bushy_Eyebrows",
        "Eyeglasses",
        "Male",
        "Mouth_Slightly_Open",
        "Mustache",
        "No_Beard",
        "Pale_Skin",
        "Young"

### Single Feature Results

| Attribute | Accuracy |
|:---------:|:--------:|
| Bald      |  75.39   |
| Bangs     |  50.61   |
| Black Hair|  33.70   |
| Blond Hair|  41.85   |
| Brown Hair|  14.61   |
| Bushy Eyebrows| 41.00|
| Male      |  87.94   |
| Mouth Open|  14.11   |
| Mustache  |  43.39   |
| No Beard  |  28.96   |
| Pale Skin |  39.03   |
| Young     |  39.77   | 

### Cumulative Feature Results

| Attribute | Accuracy |
|:---------:|:--------:|
| Male      |  87.94   |
| + Bald    |          |
| + Bangs   |          |
| + Mustache|          |
| + Blond Hair|        |