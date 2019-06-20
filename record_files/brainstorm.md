# 6/18/2019

```
ssh 10.19.129.71
```
What do I want to do? 

## Broad Goal

Predict 3D structure of proteins using just an amino acid sequence.

## Detailed Goal

Augment Mohammed AlQuraishi's [RGN](https://github.com/aqlaboratory/rgn).
* Replace the one hot encoding with a pretrained embedding representing the amino acids. (from [BERT model](https://arxiv.org/abs/1810.04805)?) Pretrain on all the amino acids.
* Replace bidirectional LSTM with universal transformer because LSTMs are slow problematic to train (vanishing/exploding gradients, its sequential nature makes it slow). It is also poor at long range dependencies.
* 

## Possible Bottlenecks 
* Training time (in the order of weeks even with 4 GPUs). Transformers should be faster to train since it is more parallelizable.

## Some Possible Ideas
* Is there a way to include physics-based principles / co-evolution in the End-to-End? (probably not).
* Use some of the suggestions from End-to-End paper.

## Things I Hope To Do
* Is there a way to decrease the training time?

## Questions
* What specs do I have for training?

Relative position?