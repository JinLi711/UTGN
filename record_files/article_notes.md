# The Illustrated Transformer

[LINK](http://jalammar.github.io/illustrated-transformer/)

2018 

* 6 stacks of encoders, 6 stacks of decoders
* in the encoder, feed forward neural network follows a self-attention layer
* in the decoder, it is self-attention, encoder-decoder attention, then a feed-forward
* create the query, key, and value vector from the word embedding by multiplying it by the corresponding matrices
* dot product the query and key vector to calculate the score
    * the score determines how much focus to place on other parts of the input sentence as we encode a word at a certain position
* normalize the scores by dividing by square root of the key dimension 
* take the softmax
* multiply each value vector by the softmax score
* sum up the weighted value vectors 
* do this for multiple heads (each with its own weights)
* heads are concatenated, then multiply it by another matrix
* trained by predicting the next word (need to mask the next word though)






# Temporal Convolutional Nets (TCNs) Take Over from RNNs for NLP Predictions

[LINK](https://www.datasciencecentral.com/profiles/blogs/temporal-convolutional-nets-tcns-take-over-from-rnns-for-nlp-pred)

2018

* TCN: much faster to train and inference than RNN
* TCN is CNN plus attention
* it is also flexible to change parameters
* not as good for adapting to transfer learning






# BERT Explained: State of the art language model for NLP

[LINK](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270)

2018

* pretrain word embeddings to be used as features for another task
* transformer reads the entire sequence at once
* 15% of the words in the sequence are predicted
    * 80% are replaced with [MASK] token
    * 10% with random word
    * 10% with orginal word
* model tries to predict the original word
* word embeddings are passed through the transformer encoder, then through a classification layer
* the output vectors is multiplied by the embedding matrix
* softmax to calculate the probability of each word in the vocabulary
* loss function only considers predictions on [MASK]
* 345 million parameters
* slower convergence from MASK prediction





# A Light Introduction to Transformer-XL

[LINK](https://medium.com/dair-ai/a-light-introduction-to-transformer-xl-be5737feb13)

2019

* biggest problem with neural nets in sequence modeling: failure to incorporate long term dependencies
* NN can suffer from context fragmentation: model lacks neccesary contextual information to predict first few symbols due to the way the context was selected
* old models do not incorporate information flow throughout segments
* transformer-xl: enable a transformer architecture to learn long-range dependencies with a recurrence mechanism without disrupting temporal coherence
* segment level recurrence mechanism: reuse previously hidden states at training time
    * reuses historical info
* uses relative positional encodings



# Transformer-XL Explained: Combining Transformers and RNNs into a State-of-the-art Language Model

[LINK](https://towardsdatascience.com/transformer-xl-explained-combining-transformers-and-rnns-into-a-state-of-the-art-language-model-c0cfe9e5a924)

2019




# The Future of Protein Science will not be Supervised

[LINK](https://moalquraishi.wordpress.com/2019/04/01/the-future-of-protein-science-will-not-be-supervised/#more-1275)

2019 

* UniRep: trains an embedding 
    * is a global model