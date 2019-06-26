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
* Relative position?
* Plug-and-play: switch around models to try (ex. bi-directional LSTM, Universal Transformer, Temporal Convolutional Neural Networks, etc)

## Things I Hope To Do
* Is there a way to decrease the training time?

## Questions
* What specs do I have for training?
* Does the RGN code do batching efficiently?

## Other 



python2.7
tensorflow-gpu==1.12.0
tensorflow==1.12.0
setproctitle


Don't remove anything that already exists.
Don't refactor code too quickly.
Options are way more limited for transformer.

Think about how I can take ideas from the RNN cell.
    GPU?
    need to add to weights and biases collection
    residual connections
    include dihedrals between layers

_recurrent_cell: don't care
understand the inputs and outputs for training and evaluating
    evaluating: no dropout
    output of outputs: [NUM_STEPS, BATCH_SIZE, RECURRENT_LAYER_SIZE]
    output of states (I think):[BATCH_SIZE, RECURRENT_LAYER_SIZE]
    I don't think the states actually matter because they aren't used.
    shape of inputs:
    Do what I need to do then transpose.

Possible causes of dead gradients:
    multiplication by 0
    non-existent gradients
    taking gradients of constant?
    division by 0
    sqrt function near 0