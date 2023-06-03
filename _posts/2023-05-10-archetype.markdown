---
layout: post
title:  "ArcheType: open-set column type annotation using large language models"
date:   2023-05-10 19:45:31 +0530
categories: ["NLP", "large language models", "big data", "dataset search"]
author: "Benjamin Feuer"
---
# ArcheType

This post describes [ArcheType](https://github.com/penfever/archetype/), a new way of doing column type annotation using large language models under the hood, which I developed under the supervision of Prof. Freire and Prof. Hegde at New York University.

## What is column type annotation?

**Column type annotation** (CTA) is an important open problem in big data literature. 

One way to conceptualize the problem is as inferring column names where none exist, typically by examining the values contained in the column and learning to apply appropriate labels to them. 

Another would be to assume that column names exist, but do not map to some unifying schema. In this case, the algorithmic task is to remap column names which are out-of-schema to their in-schema counterparts, assuming such counterparts exist.

![](../assets/auctus.png)

In this example from NYU's [Auctus](https://auctus.vida-nyu.org/) platform, the column named tpep_pickup_datetime has been automatically tagged with Text, Enumeration and DateTime column types. 

### Why should I care?

CTA is a precursor to important downstream tasks in data discovery, including establishing unionability and joinability relationships between datasets, data integration, and data cleaning. It also democratizes access to data by reducing labeling costs.

In less technical terms, a good CTA model can make searches over datasets more relevant and informative.

### Formal Analysis

We formalize CTA as follows --

Posit the existence of column(s) $C_i : \mathbb{N} \rightarrow \Sigma_\*$ in a table $T$ of column cardinality $t$, where $i$ is an index over a non-strict subset of $t$. Each column $C \in T$ is a function which maps row indices to strings, $\Sigma_C \sim \Sigma_\*$, constrained by $\forall i, 0 < |\Sigma_{C_i}| = c$. Here, $\Sigma_\*$ is the set of all possible strings. $C_i$ may include a column name, and $T$ may contain an additional metadata field; however, neither of these properties is guaranteed to exist, and so we do not include them in our analysis. The model is also given a label set $L$ of strings with cardinality $j$.

We can model $CTA : T \rightarrow L$ as a relation from tables to labels, such that $\forall C_i, \exists L_j$ and $CTA(C_i) = L_j$.

### Challenges in CTA

It sounds simple enough. So, why is CTA challenging in the real world?

* In many cases, we cannot assume the existence of a single, *correct* column type
* Identifying basic types is rarely sufficient for dataset understanding and discovery
* Fine-tuned models require lots of labeled data in order to work
* The amount of context required to solve the problem is ambiguous

### Existing CTA Methods and their Limitations

Classical approaches to the CTA problem have been largely rule-based; for instance, if $k$ is an index over $c$, and we have $\forall k, \Sigma_{C_{i_k}}$ contains the suffix "kg", we might define a rule $CTA(C_i) = \texttt{weight}$, else $CTA(C_i) = \texttt{integer}$. Such pattern matching approaches can be effective when data is clean and label sets are discrete, but when dealing with real-world data, this is rarely the case. 

Consider, for the above example, that if we have to choose between three labels, $\texttt{weight}$, $\texttt{mass}$, $\texttt{integer}$, the above rule is no longer valid. This example is not academic; all three labels exist in the real-world [SOTAB](http://webdatacommons.org/structureddata/sotab/#toc8) benchmark, for instance.

Deep learning methods such as [DoDuo](https://github.com/megagonlabs/doduo) are a substantial improvement over traditional approaches; rather than relying on a list of rules, [DoDuo](https://github.com/megagonlabs/doduo) learns them as a function of its pretraining data. By pretraining on hundreds of thousands of tables and fine-tuning on a target label set, it is able to achieve impressive results on benchmarks. 

Why, then, do we need to introduce a new method? Existing deep learning solutions for CTA have at least three major shortcomings;

* They degrade under *distribution shift* (in other words, they don't generalize well if the data they see when deployed differs much from the data they were trained on). You can see this in the image below; the DoDuo model goes from ~85% accuracy in-distribution to around 24% accuracy under shift ... on a massively simplified version of the label set.
* Label sets are fixed at training time and cannot be changed; this problem is exacerbated by problem #1, since you often have to stretch the definition of a particular model label in order to make it work with your desired labels.
* Existing methods perform best when they have access to the entire table and metadata at inference time; this is often impractical, or even impossible.

![](../assets/main_results_archetype.png)

### What is ArcheType?

**ArcheType** is a family of algorithms for column type annotation using large language models such as [ChatGPT](chat.openai.com/) and [LLAMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/). Here are some reasons we think it's an improvement on existing methods --

1. **It doesn't need access to the whole table & metadata at inference time**. On the SOTAB benchmark, we find that ArcheType can match DoDuo using just fifteen samples from the table, some summary statistics describing the column, and the table's filename. (See the table earlier in the post for those results)
2. **It works with any label set you want ... without retraining**.  Unlike prior methods, you can use ArcheType "zero-shot" -- just tell it the labels you want to apply and give it a table, and it will automatically generate a sample and produce in-schema names for each of your table's columns.
3. **You don't need to use paid APIs**. We tested ArcheType on free models; you can download them via Huggingface and host them on a single NVIDIA GPU. It's 'ready-to-deploy' in commercial settings.

### Next Steps

We believe that LLMs have enormous potential in big data environments. We're currently looking into applying it to a range of other tasks, including AutoML, dataset synthesis, property annotation (tagging), generating automatic descriptions of datasets, and conversational database queries. Stay tuned!
