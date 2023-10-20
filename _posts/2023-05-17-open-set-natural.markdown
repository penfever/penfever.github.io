---
layout: post
title:  "Weak language supervision in agriculture: opportunities and challenges"
date:   2023-05-17 19:45:31 +0530
categories: ["Computer vision", "image classification", "object detection", "agriculture", "vision-language models"]
author: "Benjamin Feuer"
---
# Image classification and object detection in agriculture

One important real-world domain for applying deep learning is agriculture. Over the last two years, under the auspices of [AIIRA](https://aiira.iastate.edu/resources/news-and-announcements/), a collaborative seven-university project sponsored by the NSF, I have had the good fortune to lead a large team of researchers from New York University, Iowa State University and the University of Arizona in developing novel tools and methods for object detection and insect classification tailored to the needs of real-world agricultural researchers and practicioners. Some of that work is described here.

## Prior work in ag-ML

Macro-organism classification (or species identification) is a computer vision problem with substantial relevance to field biologists as well as other agricultural domain experts. However, the research interests which drive the computer vision community do not always align perfectly with expert needs.

Consider the [iNaturalist](https://paperswithcode.com/dataset/inaturalist) or iNat dataset. This is a citizen scientist undertaking; images on a wide range of species are collected ad-hoc and assembled for the purposes of training species classification models. 

The iNat dataset is highly imbalanced (long-tailed) with dramatically different number of images per category. For example, the largest super-category “Plantae (Plant)” has 196,613 images from 2,101 categories; whereas the smallest super-category “Protozoa” only has 381 images from 4 categories. Furthermore, metadata is inconsistent; descriptive natural language captions for each image are not available.

From an agricultural standpoint, not all species are created equal. Insects, which are underrepresented in conventional image classification tasks such as ImageNet, are of outsized importance for ecology and crop production, as harmful and beneficial insects are inherent in the system. Agronomists, scientists and plant breeders need to identify insect species for conservation and control practices. Improper identification of species could potentially result in biased economic or action threshold estimation, leading to unnecessary application of chemicals that could harm beneficial insects, reduce profitability, and have an adverse environmental footprint. 

While manual scouting (and counting) remains the gold standard for pest identification and action threshold determination, this is a resource and (expert) labor-intensive yet critical aspect of agriculture. Additionally, genomic studies and high-throughput phenotyping rely on accurate phenotypic data including from imaging data; therefore, advances in annotation and localization of "object of interest" is necessary and an active area of current genetics and phenomics studies.

## Prior work in computer vision

### Open-Vocabulary Image Classification

OpenAI's [CLIP](https://openai.com/research/clip) was a breakthrough in open-set image classification, and has since been used for many downstream tasks, including image synthesis and object detection. By training both an image classification model and a natural language processing model in tandem, these architectures are able to learn image classification tasks without the use of manually applied ground truth labels, instead using natural language captions or tags. Thus, these architectures can be considered semi-supervised, meaning that there is considerably less work/cost involved in generating usable datasets for training, since there is little to no manual labeling of data involved. If you are unfamiliar with the architecture, we recommend reviewing it [here](https://openai.com/research/clip).
 
Robustness research by our lab and others indicated that contrastive vision-language models are considerably more robust to label noise than classifiers trained with the conventional softmax + cross-entropy loss. Additional experimental studies have demonstrated that it was possible to use a pretrained, locked image tower with far fewer samples and achieve results close to, but not on par with, vision-language models trained on semi-supervised data; this process is referred to as [LiT-Tuning](https://arxiv.org/abs/2111.07991).

### Open-Vocabulary Object Detection

Detection models focus on two loosely correlated problems: localizing objects of interest in an image, and assigning labels to them. One popular approach is a two-stage process, wherein the models detect probable object region proposals, and further finetune the bounding boxes and predict classes.

[DETIC](https://github.com/facebookresearch/Detic) presents an interesting zero-shot solution to this problem by training detection models simultaneously with object detection and image classification datasets. Formally, let $D_{det}=\{rvx_i, \{b_{i,j}, c_{i,j}\}\}$ consist of images with labelled boxes, and $D_{cls}=\{rvx_i, c_i\}$ be a classification dataset with image-level labels. Traditional detection networks consist of a two-stage detector; the first half of the network, $f_D:R^d\to \{R^m \times \left[0,1\right]\}$ outputs a set of bounding boxes and corresponding objectness scores. The second half, $f_c:R^m \to R^4 \times [c]$, takes in every proposal with an objectness score higher than a threshold and outputs a bounding box with the corresponding prediction. The networks are trained only on $D_{det}$.

Detic improves upon this by training $f_c$ on both $D_{det}$ and $D_{cls}$. The classification head in $f_c$ is also replaced with CLIP embeddings as weights to add open-set classification capabilities. Every minibatch consists of mix of samples from $D_{det}$ and $D_{cls}$. The training examples from $D_{det}$ are trained using the standard detection loss (boxwise regression and classification losses). Examples from $D_{cls}$ are assumed to have a single detected object (the largest detected box) with the image label as the box label. The model is then trained with the following loss:

$$
L(I) = \begin{cases}
     L_{RPN} + L_{Reg} + L_{cls}, ~\text{if}~I\in D_{det} \\
     \lambda L_{max-size}, ~\text{if}~I \in D_{cls}
\end{cases}
$$

Note that here, $L_{RPN}, L_{reg}$, and $L_{cls}$ refer to the training losses from [Faster RCNN](https://arxiv.org/abs/1506.01497) while $L_{max-size}$ is a cross-entropy loss with the target as the image class. 

DETIC has two advantages over traditional detectors; (1) it can learn from image classification datasets which are generally larger than detection datasets, and contain a variety of classes, and, (2) the CLIP embeddings used as the classification head allow for a far larger number of classes. Thus, contrary to standard detection models, DETIC does not require fine-tuning, and can be used for zero-shot detection with natural images.

## LiT-Tuned Models for Efficient Species Detection

In [LiT-Tuned Models for Efficient Species Detection](https://chinmayhegde.github.io/assets/papers/aaai23-lit.pdf), we focused on the problem of effectively adapting image-only datasets such as iNaturalist for use in training open-vocabulary detectors such as CLIP.

The optimal method for automatically captioning a dataset for which no captions exist is an open problem. In the original CLIP paper, Radford et al use an ensemble of 80 captions with the classname (and sometimes the object supercategory) varying each time; for example, ”A photo of a clam, a type of seafood”. Experimental results indicate that while this method is suitable for inference, it is not optimal for pretraining, owing to the homogeneity of the caption space. Furthermore, our robustness experiments indicated that ”bag of words” VL models are largely agnostic to word order and sentence syntax.

After some testing, we selected a simple method which can be deployed on any dataset for which class-level metadata exists:

* Aggregate metadata for each class
* Select a subset of metadata columns which maximize inter-class differences
* Generate image captions as 'A photo of a $\langle$ CONCAT\_METADATA $\rangle$'

The resulting dataset, **iNat-Captions**, is hosted on [HuggingFace](https://huggingface.co/ajn313/inat_captions/tree/main) and is free for public use.

### Results

We train a LiT-tuned vision language model using our lab's [VLHub](https://github.com/penfever/vlhub/) architecture, and find that despite *using a locked, pretrained image tower which has previously seen only ImageNet-1000 classes*, we are able to achieve a Top1 Accuracy of 63.28% on the 10,000-class iNat-2021 test set.

This is a surprising result, overturning much of the conventional wisdom in computer vision, which maintains that image classifiers optimize largely on the visual feature space and its architecture, with the choice of classification head typically serving as an afterthought. Rich textual representations, it seems, even without an associated grammar, can be used to adapt a computer vision classifier to accurately handle visual classes **on which it has never been trained**; the class overlap between ImageNet-21k and iNaturalist-2021 is small.

## Zero-Shot Insect Detection via Weak Language Supervision

Encouraged by these findings, [in a follow-up paper presented at the AAAI Workshop on AI for Agriculture and Food Sciences](https://chinmayhegde.github.io/assets/papers/aaai23-zs.pdf), we proceed to tackle the more challenging task of open-vocabulary object detection on agricultural data.

We curate the Insecta rank class of iNaturalist to form a new benchmark dataset, **Insecta**, of approximately 6M images consisting of 2526 agriculturally important species, and perform manual quality checking of a subset of these images to ensure their quality. 

We create a workflow tool, iNaturalist Open Download, to easily download species images from the iNaturalist Open Dataset associated with a specific taxonomy rank. We used the tool to download all images of species under the rank class Insecta from the iNaturalist Open Dataset for downstream annotation, curation and use in our model. We choose to only use images identified as “research” quality grade under the iNaturalist framework, which indicates that the labeling inspection for the image is more rigorous than standard and has multiple agreeing identifications at the species level. This results in a total of 13,271,072 images across 95,399 different insect species at the time of writing. The images have a maximum resolution of 1024x1024, in .jpg/.jpeg format and total 5.7 terabytes. Among the 95,399 insect species, we select 2526 species which have been reported to be the most agriculturally important species. This subset of insect classes contribute to 6 million images in total.

We further demonstrate that using zero-shot object detectors such as DETIC, inference can be performed by simply specifying the category/class using a new natural language description. A single (universal) natural language prompt provides highly accurate bounding box for a very large dataset of diverse insect-pest images.

![](../assets/insecta_samples_01.png)

![](../assets/insecta_samples_02.png)

## Future Work

Although significant progress has been made, there remain important tasks for which the best currently existing computer vision models are ill-suited, such as counting instances of pests at a distance and localizing particular instances using natural language commands.

There also exist opportunities beyond the purview of computer vision; large language models hold considerable promise for zero-shot scientific annotation and species description, as well as visual question answering.

## Acknowledgements

I wish to gratefully acknowledge my many collaborators on this line of work, including Ameya Joshi, Minsu Cho, Kewal Jani, Shivani Chiranjeevi, Andre Nakkab, Zi Kang Deng, Aditya Balu, Asheesh K. Singh, Soumik Sarkar, Nirav Merchant, Arti Singh, Baskar Ganapathysubramanian and Chinmay Hegde.
