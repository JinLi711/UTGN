# Learning protein sequence embeddings using information from structure

[LINK](https://openreview.net/forum?id=SygLehCqtm)

2018







# Training Tips for the Transformer Model

[LINK](https://arxiv.org/abs/1804.00247)

2018







# Self-Attention with Relative Position Representations

[LINK](https://arxiv.org/abs/1803.02155)

2018







# UNIVERSAL TRANSFORMERS

2018

[LINK](https://arxiv.org/pdf/1807.03819v3.pdf)

READ THIS





# End-to-end differentiable learning of protein structure

[LINK](https://www.biorxiv.org/content/10.1101/265231v2)

2018

```
@article{AlQuraishi2019EndtoEndDL,
  title={End-to-End Differentiable Learning of Protein Structure.},
  author={Mohammed AlQuraishi},
  journal={Cell systems},
  year={2019}
}
```

## Abstract

* uses Bi-directional LSTM
* parameterized local protein structures with torsional angle
* coupled local protein structure to its global representation with recurrent geometric units
* used differential loss function

## Introduction

* introduces building blocks for constructing end-to-end differentiable model of protein structure
* want to try to predict protein structure without co-evolutionary information

## Results

### Recurrent Geometric Networks

* input: sequence of amino acids and PSSMs of a protein
* output: 3D structure
* model consists of three stages: computation, geometry, and assessment. Call the model Recurrent Geometric Networks (RGN)
    * pass through a sequence, output is converted into angles, which are then converted to coordinates
* assumes fixed bond length and angle


## Discussion

### Immediate Extensions

* incorporate co-evolutionary information as priors
* incorporate templates possibly with a confidence score
* predict side-chain conformations

<img src="images/Recurrent Geometric Networks.png" height="500" width="800">










# ProteinNet: a standardized data set for machine learning of protein structure

[LINK](https://arxiv.org/pdf/1902.00249.pdf)

2019

```
@inproceedings{AlQuraishi2019ProteinNetAS,
  title={ProteinNet: a standardized data set for machine learning of protein structure},
  author={Mohammed AlQuraishi},
  booktitle={BMC Bioinformatics},
  year={2019}
}
```

## Abstract

* lack of standardized dataset for ML on protein structure prediction
* used evolution-based distance metrics to create a difficult developement set

## Introduction

* need to deal with missing residues, fragmentations, non-contiguous polypeptide chains
* MSAs are available for every structure
* split into train, validation, and test set
	* can be difficult because data for proteins is not i.i.d

## Methods

### Structures and sequences
* test set is CASP, train set is avaliable protein structures before the CASP
* subset of training is set aside for validation at different sequence identity thresholds
* CASP has template based modeling (TBM) (there already exists a similar protein) and free modeling
* excluded structures with <2 residues or if >90% of the structure is not resolved
* mask records to indicate which residues are missing
* Multiple logical chains that correspond to a single physical polypeptide chain are combined
* physically distinct polypeptide chains are seperated
* For chains with multiple models, only the first one is kept
* if aa is chemically modified or if its identity is unknown, use PSSM to find most probable residue
* if PSSM contains more than 3 adjacent residues with 0 info content, then its corresponding sequence / structure is dropped


## Results

*  34,557 to 104,059 structures








# Biological Structure and Function Emerge from Scaling Unsupervised Learning to 250 Million Protein Sequences

[LINK](https://www.biorxiv.org/content/10.1101/622803v1.abstract)

2019

```
@article {Rives622803,
	author = {Rives, Alexander and Goyal, Siddharth and Meier, Joshua and Guo, Demi and Ott, Myle and Zitnick, C. Lawrence and Ma, Jerry and Fergus, Rob},
	title = {Biological Structure and Function Emerge from Scaling Unsupervised Learning to 250 Million Protein Sequences},
	elocation-id = {622803},
	year = {2019},
	doi = {10.1101/622803},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2019/04/29/622803},
	eprint = {https://www.biorxiv.org/content/early/2019/04/29/622803.full.pdf},
	journal = {bioRxiv}
}
```

## Abstract

* 250 million sequences
* maps sequences to representations


## Introduction

* self-supervision: given a sequence, predict the missing elements in the sequence
* data is from: Uniparc database, created in 2007


## Background

* aa sequences only hav 25 elements, can be comparable to character level representations
* aa sequences are much longer than word sentences.


## SCALING LANGUAGE MODELS TO 250 MILLION DIVERSE PROTEIN SEQUENCES

* bidirectional transformer
* mask a amino acid and use the context to predict the mask



## MULTI-SCALE ORGANIZATION IN SEQUENCE REPRESENTATIONS

* learned representation encodes underlying factors that influence sequence variations in the data.

## Discussion

* 700M parameters still underfits







# PAY LESS ATTENTION WITH LIGHTWEIGHT AND DYNAMIC CONVOLUTIONS

[LINK](https://arxiv.org/pdf/1901.10430v2.pdf)

2019









# STCN: STOCHASTIC TEMPORAL CONVOLUTIONAL NETWORKS

[LINK](https://arxiv.org/pdf/1902.06568.pdf)

2019