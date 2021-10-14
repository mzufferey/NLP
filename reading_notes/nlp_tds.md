### Some examples of applying BERT in specific domain](https://towardsdatascience.com/how-to-apply-bert-in-scientific-domain-2d9db0480bd9)

 SciBERT which is based on BERT to address the performance on scientific data. It uses a pre-trained model from BERT and fine-tune  contextualized embeddings by using scientific publications which  including 18% papers from computer science domain and 82% from the broad biomedical domain.

generic pretrained NLP model may not work very well in specific domain  data. Therefore, they fine-tuned BERT to be BioBERT and 0.51% ~ 9.61%  absolute improvement in biomedical’s NER, relation extraction and  question answering NLP tasks

Both SciBERT and BioBERT also introduce domain specific data for pre-training.

![img](https://miro.medium.com/max/560/1*MhNie2aPLIid3hq_uh0qqg.png)

Both SciBERT and BioBERT follow BERT model architecture which is **multi  bidirectional transformer** and learning text representation by predicting masked token and next sentence.

 A sequence of tokens will be transform to token embeddings, segment  embeddings and position embeddings. 

**Token embeddings** refers to  contextualized word embeddings;

**segment embeddings** only include 2  embeddings which are either 0 or 1 to represent first sentence and  second sentence

**position embeddings** stores the token position relative  to the sequence. 

### [How BERT leverage attention mechanism and transformer to learn word contextual relations](https://towardsdatascience.com/how-bert-leverage-attention-mechanism-and-transformer-to-learn-word-contextual-relations-5bbee1b6dbdb)

a new state-of-the-art NLP paper is released by Google. They call this  approach as BERT (Bidirectional Encoder Representations from  Transformers).

 **transformer** architecture to learn the text representations. 

BERT uses **bidirectional transformer** (both  left-to-right and right-to-left direction) rather than dictional  transformer (left-to-right direction). 

BERT uses **bidirectional language model** to learn the text representations 

BERT uses deep neural network

##### Input Representation

 3 embeddings to compute the input representations

1. token embeddings;  
   * general word embeddings
   * uses vector to represent token (or word) (see [here](https://towardsdatascience.com/3-silver-bullets-of-word-embedding-in-nlp-10fa8f50cc5a) for detail)

2. segment embeddings 
   * sentence embeddings 
   * If input includes 2 sentence,  corresponding sentence embeddings will be assigned to particular words.  
   * If input only include 1 sentence, one and only one sentence embeddings will be used. 
   * learnt before computing BERT (see [here](https://towardsdatascience.com/transforming-text-to-sentence-embeddings-layer-via-some-thoughts-b77bed60822c) for detail)
3. position embeddings
   *  token sequence of input
   * even if there are 2 sentences, position will be accumulated.

“CLS” is the reserved token to represent the start of sequence while “SEP” separate segment (or sentence)

##### Pre-Training Tasks

First pre-training task is **masked language model** while the second task is **predicting next sentence**.

*Masked language model* (masked LM)

bidirectional rather than traditional directional as a  pre-training objective

If using traditional approach to train a  bidirectional model, each word will able to see “itself” indirectly

Masked Language Model (MLM) approach instead: by masking  some tokens randomly, using other token to predicted those masked token  to learn the representations

Unlike other approaches, predicts masked token rather than entire input.

the experiment picks 15% of token randomly to be replaced.

downsides:

1.  MASK token (actual  token will be replaced by this token) will never be seen in fine-tuning  stage and actual prediction. Therefore, Devlin et al, the selected token for masking will not alway be masked but
   * A: 80% of time, it will be replaced by [MASK] token
   * B: 10% of time, it will be replaced by other actual token
   * C: 10% of time, it will be keep as original.

2. only 15% token is masked (predicted) per batch, a longer time will take for training.

*Next Sentence Prediction*

 predict next sentence

overcome the issue of first task as it cannot learn the relationship between sentences

simple objective: only classifying whether second sentence is next sentence or not

##### Training the model

2 phases training: 1) using generic data set to perform first training; 2) fine tuning by providing domain specific data set.

*Pre-training phase*

sentences retrieved from BooksCorpus (800M words) and English Wikipedia (2500M words)

step done by Google research team and we can leverage this pre-trained model to further fine tuning model based on own data.

*Fine-tuning phase*

Only some model hyperparameters are changed such as batch size, learning rate and number of training epochs, most mode hyperparameters are kept as same in pre-training phase.

Fine-tuning procedure is different and it depends on downstream tasks.

* classification: 
  * [CLS] token feed as the final hidden state
  * Label  (C) probabilities computed with a softmax
  * fine-tuned to maximize the log-probability of the correct label.
* Named Entity Recognition: 
  * Final hidden representation of token feed into the  classification layer. 
  * Surrounding words considered on the  prediction. 
  * the classification only focus on the token  itself and no Conditional Random Field (CRF).

### [Transformers from scratch](http://peterbloem.nl/blog/transformers)

##### Self-attention

The fundamental operation of any transformer architecture is the *self-attention operation*.

Self-attention is a **sequence-to-sequence operation**: a sequence of vectors goes in, and a sequence of vectors comes out. 

To produce output vector 𝐲i, the self attention operation simply takes *a **weighted average over all the input vectors***

𝐲i=∑jwij𝐱j.

Where j indexes over the whole sequence and the weights sum to one over all j. The weight wij is not a parameter, as in a normal neural net, but it is *derived* from a function over 𝐱i and 𝐱j. The simplest option for this function is the dot product:

The dot product gives us a value anywhere between negative and positive infinity, so we apply a softmax to map the values to [0,1] and to ensure that they sum to 1 over the whole sequence:

wij=exp w′ij∑jexp w′ij

And that’s the basic operation of self attention.

this is **the only operation in the whole architecture that propagates information *between* vectors**.

Every other operation in the transformer is applied to each vector in the input sequence without interactions between vectors.

##### Understanding why self-attention works

example: customer movie features matching movie features, dot product to get the match between them; Annotating a database of millions of movies is very costly, and  annotating users with their likes and dislikes is pretty much impossible;  instead is that we make the movie features and user features ***parameters*** of the model. We then ask users for a small number of movies that they  like and **we optimize the user features and movie features so that their  dot product matches the known likes**

Even though we don’t tell the model what any of the features should mean, in practice, it turns out that after training the features do actually reflect meaningful semantics about the movie content

This is the basic principle at work in the self-attention

 Let’s say we are faced with a sequence of words. To apply self-attention, 

* **embedding layer**: 
  * assign each word t in our vocabulary an *embedding vector* 𝐯t (the values of which we’ll learn)
  * turns the word sequence into the vector sequence
* **self-attention layer**
  * input is the embedding vector, the output is another sequence of vectors 
  * which element of the output vector is a weighted sum over all the embedding vectors in the first sequence, weighted by their (normalized) dot-product with the element of the embedding vector

we are *learning* what the values in the embedding vector should be, how "related" two words are is entirely determined by the task.

The **dot product** expresses how related two vectors in the input sequence  are, with “**related**” defined by the learning task, and the output vectors are **weighted sums over the whole input sequence**, with the **weights  determined by these dot products**.

following properties, which are unusual for a sequence-to-sequence operation

* There are **no parameters** 
* Self attention sees its input as a *set*, not a sequence. If we  permute the input sequence, the output sequence will be exactly the  same, except permuted also (i.e. **self-attention is *permutation  equivariant***); self-attention by itself actually ignores the sequential nature of the input

##### basic self-attention implementation

​			

```
import torch
import torch.nn.functional as F

# assume we have some tensor x with size (b, t, k)
x = ...

# The set of all raw dot products w′ij forms a matrix, which we can compute simply by multiplying 𝐗 by its transpose: 

raw_weights = torch.bmm(x, x.transpose(1, 2))
# - torch.bmm is a batched matrix multiplication. It 
#   applies matrix multiplication over batches of 
#   matrices

#  turn the raw weights w′ij into positive values that sum to one, we apply a *row-wise* softmax:
weights = F.softmax(raw_weights, dim=2)

# Finally, to compute the output sequence, we just multiply the weight matrix by 𝐗. This results in a batch of output matrices 𝐘 of size `(b, t, k)` whose rows are weighted sums over the rows of 𝐗.
y = torch.bmm(weights, x)
```

##### Additional tricks

The actual self-attention used in modern transformers relies on three additional tricks.

*1) Queries, keys and values*

Every input vector 𝐱i is used in 3 different ways in the self attention operation:

1. compared to every other vector to establish the weights for its own output 𝐲i (**query**)
2. compared to every other vector to establish the weights for the output  of the j-th vector 𝐲j (**key**)
3. used as part of the weighted sum to compute each output vector once the weights have been established (**value**)

basic self-attention: each input vector must play all three roles

make easier: by deriving new  vectors for each role, by applying a linear transformation to the  original input vector. In other words, we add three k×k weight matrices 𝐖q, 𝐖k,𝐖v and compute three linear transformations of each xi, for the three different parts of the self attention:

This gives the self-attention layer some controllable parameters, and allows it to modify the incoming vectors to suit the three roles they  must play.

*2) Scaling the dot product*

The softmax function can be sensitive to very large input values.  These kill the gradient, and slow down learning, or cause it to stop  altogether. Since the average value of the  dot product grows with the  embedding dimension k, it helps to scale the dot product back a little to stop the inputs to the softmax function from growing too large

3) Multi-head attention

account for the fact that a word can mean different things to different neighbours

ex: mary,gave,roses,to,susan -> different relations for gave. mary expresses who’s doing the giving, roses expresses what’s being given, and susan expresses who the recipient is.

In a single self-attention operation, all this information just gets summed together. If Susan gave Mary the roses instead, the output vector 𝐲gave would be the same, even though the meaning has changed.

give the self attention greater power of discrimination, by combining several self attention mechanisms (which we'll index with r), each with different matrices 𝐖rq, 𝐖rk,𝐖rv. These are called **attention heads**.

For input 𝐱i each attention head produces a different output vector 𝐲ri. We **concatenate these, and pass them through a linear transformation** to reduce the dimension back to k.

<u>Efficient multi-head self-attention</u>. The simplest way to understand **multi-head self-attention** is to see it as a small number of copies of the self-attention mechanism applied in parallel, each with their own key, value and query transformation. This works well, but for R heads, the self-attention operation is R times as slow.

there is a way to implement multi-head self-attention so that it is roughly as fast as the single-head version, but we still get the benefit of having different attention matrices in parallel. To accomplish this, we **cut each incoming vector into chunks**: if the input vector has 256 dimensions, and we have 8 attention heads, we cut it into 8 chunks of 32 dimensions. For each chunk, we **generate keys, values and queries of 32 dimensions each**. This means that the matrices 𝐖rq, 𝐖rk,𝐖rv are all 32×32.

##### Building transformers

A transformer is not just a self-attention layer, it is an **architecture**. It’s not quite clear what does and doesn’t qualify as a transformer, but here we’ll use the following definition:

a **transformer** = any architecture designed to **process a connected set of units**—such as the tokens in a sequence or the pixels in an image—where the only **interaction between units is through self-attention**. 

standard approach for how to build self-attention layers up into a larger network. 1st step: wrap the self-attention into a block that we can repeat.

*The transformer block*

some variations on how to build a basic transformer block, but most of them are structured roughly like this:

![img](http://peterbloem.nl/files/transformers/transformer-block.svg)

the block applies, in sequence: 

* a self attention layer
* layer normalization
* a feed forward layer (a single MLP applied independently to each vector)
* another layer normalization.
* residual connections are added around both, before the normalization. 

The order of the  various components is not set in stone; the important thing is to **combine self-attention with a local feedforward, and to add  normalization and residual connections**.

Normalization and residual connections are standard tricks used to help  deep neural networks train faster and more accurately. The layer  normalization is applied over the embedding dimension only.

*Classification transformer*

The simplest transformer = a sequence classifier. 

The heart of the architecture will simply be a large chain of transformer blocks. 

All we need to do is work out how to feed it the input sequences, and how to transform the final output sequence into a a single classification.



The most common way to build a **sequence classifier** out of  sequence-to-sequence layers, is to apply **global average pooling** to the  final output sequence, and to **map the result to a softmaxed class  vector**.

![img](http://peterbloem.nl/files/transformers/classifier.svg) Overview of a simple sequence classification transformer. **The output sequence is averaged to produce a single vector representing the whole sequence. This vector is projected down to a vector with one element per class and softmaxed  to produce probabilities**.



##### Input: using the positions

embedding layer to represent the words.

stacking permutation equivariant layers, and the final global average pooling is permutation *in*variant, so the network as a whole is also permutation invariant.

if we shuffle up the words in the sentence, we get the exact same classification, whatever weights we learn

but we want our state-of-the-art language model to have at least some sensitivity to word order,

solution: create a second vector of equal length, that represents the position of  the word in the current sentence, and add this to the word embedding; 2 options

1. **position embeddings**:  embed the positions like we did the words. 
   * drawback: we have to see sequences of every length during training, otherwise the relevant position embeddings don't get trained
   * benefit: works pretty well, and easy to implement
2. **position encodings**: work in the same way as embeddings, except that we don't *learn* the position vectors, we just choose some function to map the positions to real valued vectors, and let the network figure out how to interpret these encodings
   * benefit: for a well  chosen function, the network should be able to deal with sequences that  are longer than those it's seen during training (it's unlikely to  perform well on them, but at least we can check)
   * drawbacks: the choice of encoding function is a complicated hyperparameter, and it complicates the implementation a little.

##### Text generation transformer

try an **autoregressive model**; train a ***character* level transformer to predict the next character** in a sequence.

give the sequence-to-sequence model a sequence, and ask it to  predict the next character at each point in the sequence; the target output is the same sequence shifted one character to  the left:

![img](http://peterbloem.nl/files/transformers/generator.svg)

With RNNs this is all we need to do, since they cannot look forward into the input sequence: output i depends only on inputs 0 to i. 

**With a transformer, the output depends on the entire input sequence**, so prediction of the next character becomes vacuously easy, just retrieve  it from the input.

To use **self-attention as an autoregressive model**, need to  **ensure that it cannot look forward into the sequence**. We do this by  **applying a mask to the matrix of dot products**, before the softmax is  applied. This mask **disables all elements above the diagonal of the  matrix**.



##### Design considerations

The main point of the transformer was to overcome the problems of the previous state-of-the-art architecture, the RNN 

 big weakness here is the **recurrent connection**

*  this allows information to propagate along the sequence, 
* but means that we cannot compute the cell at time step i until we’ve computed the cell at timestep i−1.

contrasts with 1D convolution: 

* each vector can be computed in parallel with  every other output vector; much faster
* drawback: limited in modeling *long range dependencies*. In one convolution layer,  only words that are closer together than the kernel size can interact  with each other. For longer dependence we need to stack many  convolutions.

The transformer is an attempt to capture the best of both worlds.

* can model dependencies over the whole range of the input sequence  just as easily as they can for words that are next to each other (in  fact, without the position vectors, they can’t even tell the  difference)
* no recurrent connections, so the whole  model can be computed in a very efficient feedforward fashion.

The rest of the design of the transformer is based primarily on one  consideration: depth. Most choices follow from the desire to train big  stacks of transformer blocks. 

only  2  places in the transformer where non-linearities occur: 

1. the softmax in  the self-attention 
2. the ReLU in the feedforward layer. 

The rest of the model is entirely composed of linear transformations, which perfectly preserve the gradient.

layer normalization likely also nonlinear, but this nonlinearity helps to keep the gradient stable as it  propagates back down the network.

##### Why is it called self-*attention*?

Before self-attention, sequence models consisted  mostly of recurrent networks or convolutions stacked together. 

it was discovered that these models could be helped by adding ***attention mechanisms***: instead of feeding the output sequence of the previous layer directly  to the input of the next, **an intermediate mechanism was introduced, that decided which elements of the input were relevant for a particular word of the output**.

The general mechanism was as follows.

*  call the input the **values**
* some (trainable) mechanism assigns a **key** to each value
* then to each output, some other mechanism assigns a **query**.

These names derive from the datastructure of a **key-value store**. In  that case we expect only one item in our store to have a key that  matches the query, which is returned when the query is executed.  

Attention is a softened version of this: ***every* key in the store matches the query to some extent. All are returned, and we take a sum,  weighted by the extent to which each key matches the query**.

The great breakthrough of **self-attention** was that **attention by itself is a strong enough mechanism to do all the learning**; **The key, query and value are all the same vectors (with minor linear transformations).** They ***attend to themselves*** and **stacking such self-attention provides sufficient nonlinearity and  representational power** to learn very complicated functions.

##### BERT 

BERT was one of the first models to show that transformers could reach human-level performance on a variety of language based tasks: question answering, sentiment classification or classifying whether two sentences naturally follow one another.

BERT consists of a simple **stack of transformer blocks**

pre-trained on a large general-domain corpus consisting of 800M words from English books (modern work, from unpublished authors), and 2.5B words of text from English Wikipedia articles (without markup).

**Pretraining** is done through two tasks:

1. **Masking**
   * A certain number of  words in the input sequence are: masked out, replaced with a random word or kept as is. 
   * The model is then asked to predict, for these words, what the original words were
   * **the model doesn't need to predict the entire denoised sentence, just the modified words**
   * Since the model doesn't know which words it will be asked about, it learns a representation for every word in the sequence.
2. Next sequence classification
   * 2 sequences of about 256 words are sampled that either (a) follow each other directly in the corpus, or (b) are both taken from random places. 
   * The model must then predict whether a or b is the case.

BERT uses **WordPiece tokenization**

* somewhere in **between word-level and character level sequences.** 
* breaks words like walking up into the tokens walk and ##ing -> allows the model to make some inferences based on word structure: two verbs ending in -ing have similar grammatical functions, and two verbs starting with walk- have similar semantic function.

The input is **prepended with a special <cls> token**. **The output vector corresponding to this token is used as a sentence representation in sequence classification tasks** like the next sentence classification (as opposed to the global average pooling over all vectors that we used in our classification model above).

After pretraining, a **single task-specific layer** is placed after the body of transformer blocks, which maps the general purpose representation to a task specific output. 

For classification tasks, this simply maps the first output token to softmax probabilities over the classes. 

For more complex tasks, a final sequence-to-sequence layer is designed specifically for the task.

The whole model is then re-trained to finetune the model for the specific task at hand.

In an ablation experiment, the authors show that **the largest improvement as compared to previous models comes from the bidirectional nature of BERT**. 

* previous models like GPT used an **autoregressive mask**, which allowed attention only over previous tokens. 
* in BERT **all attention is over the whole sequence** is the main cause of the improved performance. This is why the B in BERT stands for "bidirectional".

The largest BERT model uses 24 transformer blocks, an embedding dimension of 1024 and 16 attention heads, resulting in 340M parameters.

##### Sparse transformers

 tackle the problem of quadratic memory use  head-on. I**nstead of computing a dense matrix of attention weights (which grows quadratically), they compute the self-attention only for  particular pairs of input tokens**, resulting in a ***sparse* attention matrix**, with only n*√n explicit elements.

* benefit: 
  * allows models with very large context sizes, for instance for  generative modeling over images, with large dependencies between pixels; allow to train transformers with very large  sequence lengths
  *  also allows a very elegant way  of designing an inductive bias. We take our input as a collection of  units (words, characters, pixels in an image, nodes in a graph) and we  specify, through the sparsity of the attention matrix, which units we  believe to be related. The rest is just a matter of building the  transformer up as deep as it will go and seeing if it trains.
* tradeoff: sparsity structure is not learned, so by the  choice of sparse matrix, we are disabling some interactions between  input tokens that might otherwise have been useful. However, two units  that are not directly related may still interact in higher layers of the transformer (similar to the way a convolutional net builds up a larger  receptive field with more



### [Fleuret NLP lectures](https://fleuret.org/dlc/)

A common word embedding is the Continuous Bag of Words (CBOW) version
of **word2vec** (Mikolov et al., 2013a).
In this model, the embedding vectors are chosen so that a word can be
[linearly] predicted from the sum of the embeddings of words around it.

An alternative algorithm is the **skip-gram model**, which optimizes the
embedding so that a word can be predicted by any individual word in its context
(Mikolov et al., 2013a).

The main benefit of word embeddings is that they are trained with unsupervised
corpora, hence possibly extremely large.

in all the operations we have seen such as fully connected layers, convolutions,
or poolings, **the contribution of a value in the input tensor to a value in the**
**output tensor is entirely driven by their [relative] locations [in the tensor].**

**Attention mechanisms** aggregate features with an importance score that

* depends on the feature themselves, not on their positions in the tensor,
* relax locality constraints

Attention mechanisms **modulate dynamically the weighting of different parts of**
**a signal and allow the representation and allocation of information channels to**
**be dependent on the activations themselves**.

While they were developed to equip deep-learning models with memory-like
modules (Graves et al., 2014), their main use now is to **provide long-term**
**dependency for sequence-to-sequence translation** (Vaswani et al., 2017).



##### attention mechanisms

the simplest form of attention is **content-based attention**.

which differs from **context attention**, 

the most classical version of attention is a **context-attention with a dot-product**
**for attention function**

 using the terminology of Graves et al. (2014), attention is an **averaging of**
**values associated to keys matching a query**. Hence the keys used for
computing attention and the values to average are different quantities

attention layer to equip the model with the **ability to combine**
**information from parts of the signal that it actively identifies as relevant**.



##### Transformer networks

Vaswani et al. (2017) proposed to go one step further: instead of using
attention mechanisms as a supplement to standard convolutional and recurrent
operations, they designed **a model combining only attention layers**.

They designed this “**transformer**” for a sequence-to-sequence translation task,
but it is currently key to state-of-the-art approaches across NLP tasks

they first introduce a multi-head attention module

Their complete model is composed of:

* An **encoder** that combines N = 6 modules each composed of a **multi-head**
  **attention sub-module**, and a **[per-component] one hidden-layer MLP**, with
  residual pass-through and layer normalization.
* A **decoder** with a similar structure, but with **causal attention layers** to allow
  for regression training, and **additional attention layers** that attend to the
  layers of the encoder.

Positional information is provided through an **additive positional encoding**

The **Universal Transformer** (Dehghani et al., 2018) is a similar model where **all**
**the blocks are identical**, resulting in a recurrent model that iterates over
consecutive revisions of the representation instead of positions.
Additionally the **number of steps is modulated per position dynamically**.

transformer networks were introduced for translation, and trained with a
supervised procedure, from pairs of sentences

 they can be trained in a unsupervised
manner, for auto-regression or as denoising auto-encoders, from very large
data-sets, and fine-tuned on supervised tasks with small data-sets

BERT (Bidirectional Encoder Representation from Transformers) is a transformer pre-trained with:

* **Masked Language Model** (MLM), that consists in predicting [15% of]
  words which have been replaced with a “MASK” token.
* **Next Sentence Prediction** (NSP), which consists in predicting if a certain
  sentence follows the current one.

It is then fine-tuned on multiple NLP tasks.

##### Attention in computer vision

They insert “non-local blocks” in residual architectures and get improvements
on both video and images classification.

### [BERT Explained: A Complete Guide with Theory and Tutorial](https://towardsml.com/2019/09/17/bert-explained-a-complete-guide-with-theory-and-tutorial/)

**pre-training** = raining general purpose language representation models using the  enormous piles of unannotated text on the web 

These general purpose pre-trained models can then be ***fine-tuned\*** on smaller task-specific datasets, e.g., when working with problems like question answering and sentiment analysis.

BERT is a recent addition to these techniques for NLP pre-training

we can either use the BERT models to extract high quality language  features from our text data, or we can fine-tune these models on a  specific task

In the pre-BERT world, **one-directional approach** works well for generating sentences — we can  predict the next word, append that to the sequence, then predict the  next to next word until we have a complete sentence.

 BERT, a language model which is **bidirectionally trained** (this is also its key technical innovation). This means we can now have a  deeper sense of language context and flow compared to the  single-direction language models.

Instead of predicting the next word in a sequence, BERT makes use of a novel technique called **Masked LM** (MLM): it randomly masks words in the sentence and then it tries to predict them. 

**Masking**  means that the model looks in both directions and it uses the full  context of the sentence, both left and right surroundings, in order to  predict the masked word. 

Unlike the previous language models, it takes  **both the previous and next tokens into account** at the **same time.** 

The existing combined left-to-right and right-to-left LSTM based models  were missing this “same-time part”. (It might be more accurate to say  that BERT is non-directional though.)

Pre-trained language representations can either be

* **context-free** (e.g. word2vec)
  * generate a single word embedding representation (a vector of numbers) for each word in the vocabulary.
  * e.g.  “bank” would have the same context-free representation in “bank account” and “bank of the river"
* **context-based**
  * generate a representation of each word that is based on the other words in the sentence
  * **unidirectional** 
    * e.g.  in  “I accessed the bank account,” represent “bank” based on “I accessed the” 
  * *bidirectional ** (e.g. BERT)
    * e.g. represents “bank” using both its previous and next context — “I accessed the … account”

Moreover, BERT is based on the **[Transformer model architecture](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html),** instead of LSTMs. 

A **Transformer** works by performing a small, constant number of steps. 

* In  each step, it applies an **attention mechanism** to understand relationships between all words in a sentence, regardless of their respective  position. 
* For example, given the sentence, “I arrived at the bank after crossing the river”, to determine that the word “bank” refers to the  shore of a river and not a financial institution, the Transformer can  **learn to immediately pay attention to the word “river” and make this  decision in just one step.**

BERT relies on a **Transformer** (the **attention mechanism that learns contextual relationships between words** in a text). A basic Transformer consists of an **encoder** to read the text input and a **decoder** to produce a prediction for the task. 

Since **BERT’s goal is to generate a language  representation model**, it only needs the **encoder** part. 

The **input** to the  encoder for BERT is a **sequence of tokens**, which are first converted into **vectors** and then processed in the neural network. But before processing can start, BERT needs the input to be massaged and decorated with some  extra metadata:

1.  **Token embeddings**:
   * A [CLS] token is added to the input word tokens **at the beginning of the first sentence** 
   * a [SEP] token is inserted at the end of each  sentence.
2. **Segment embeddings**: 
   * A marker **indicating Sentence A or Sentence B** is added to each token. This allows the encoder to distinguish between sentences.
3. **Positional embeddings**: 
   * A positional embedding is added to each token to **indicate its position in the sentence**.

Essentially, t**he Transformer stacks a layer that maps sequences to sequences**, so the output is also a sequence of vectors with a 1:1  correspondence between input and output tokens at the same index. 

BERT does not try to predict the next word in the  sentence. 

Training makes use of the following two strategies:

**1. Masked LM (MLM)**

Procedure:

* Randomly mask out 15% of the words in the  input — replacing them with a [MASK] token 
* run the entire sequence  through the BERT attention based encoder and then predict only the  masked words, based on the context provided by the other non-masked  words in the sequence. 

problem with this naive  masking approach: the model only tries to predict when the [MASK]  token is present in the input, while we want the model to try to predict the correct tokens regardless of what token is present in the input. To deal with this issue, out of the 15% of the tokens selected for  masking:

- 80% of the tokens actually replaced with the token [MASK].
- 10% replaced with a random token.
- 10% left unchanged.

While training the BERT loss function **considers only the prediction of the  masked tokens** and ignores the prediction of the non-masked ones. This  results in a model that **converges much more slowly** than left-to-right or right-to-left models.

2. **Next Sentence Prediction (NSP)**

In order to **understand relationship between two sentences**, BERT training process also uses next sentence prediction. 

A pre-trained model with this kind of understanding is **relevant for tasks like question answering**. 

During training the model gets as i**nput pairs of sentences and it learns to predict if the second sentence is the next sentence in the original text** as well.

As we have seen earlier, BERT separates sentences with a special [SEP] token. During training the model is fed with two input sentences at a time such that:

* 50% of the time the second sentence comes after the first one.
* 50% of the time it is a a random sentence from the full corpus.

BERT is then required to **predict whether the second sentence is random or not**, with the assumption that the random sentence will be disconnected from the first sentence (predict label "IsNext" or "NotNext")

To predict if the second sentence is connected to the first one or not,  basically the complete input sequence goes through the Transformer based model, the output of the [CLS] token is transformed into a 2×1 shaped  vector using a simple classification layer, and the IsNext-Label is  assigned using softmax.

The model is **trained with both Masked LM and Next Sentence Prediction  together**. This is to **minimize the combined loss function** of the two  strategies — *“together is better”*.

##### Architecture

There are **four types of pre-trained versions of BERT** depending on the scale of the model architecture:

**`BERT-Base`**: 12-layer, 768-hidden-nodes, 12-attention-heads, 110M parameters
 **`BERT-Large`**: 24-layer, 1024-hidden-nodes, 16-attention-heads, 340M parameters


if we want to fine-tune the original model based on our own dataset, we  can do so by just adding a single layer on top of the core model.

For example, say we are creating **a question answering application**. In essence question answering is just a **prediction task** — on receiving a question as input, the goal of the application is to identify the  right answer from some corpus. 

So, given a question and a context  paragraph, **the model predicts a start and an end token from the  paragraph that most likely answers the question** -> BERT can be trained by learning **two extra  vectors that mark the beginning and the end of the answer**.

Just like sentence pair tasks, **the question becomes the first  sentence and paragraph the second sentence in the input sequence**.  However, this time there are two new parameters learned during  fine-tuning: a **start vector** and an **end vector.**

in case we want to do fine-tuning, we need to **transform our input into  the specific format** that was used for pre-training the core BERT models, e.g., we would need to add special tokens to mark the beginning ([CLS]) and separation/end of sentences ([SEP]) and segment IDs used to  distinguish different sentences — convert the data into features that  BERT uses.

, we can also do custom **fine tuning** by **creating a single new layer** **trained to adapt BERT** to our sentiment task (or any other task). 

we need to preprocess our data so that it matches the data  BERT was trained on. For this, we'll need to do a couple of things (but  don't worry--this is also included in the Python library):

* Lowercase our text (if we're using a BERT lowercase model)

* Tokenize it (i.e. "sally says hi" -> ["sally", "says", "hi"]) 
* Break words into WordPieces (i.e. "calling" -> ["call", "##ing"]) 
* Map our words to indexes using a vocab file that BERT provides 
* Add special "CLS" and "SEP" tokens (see the [readme](https://github.com/google-research/bert)) 
* Append "index" and "segment" tokens to each input (see the [BERT paper](https://arxiv.org/pdf/1810.04805.pdf)) 

First, it loads the BERT tf hub module again (this time to extract the computation graph). Next, it creates a single new layer that will be trained to adapt BERT to our sentiment task (i.e. classifying whether a movie review is positive or negative). This strategy of using a mostly trained model is called fine-tuning.

### [BERT : Le "Transformer model" qui s’entraîne et qui représente](https://lesdieuxducode.com/blog/2019/4/bert--le-transformer-model-qui-sentraine-et-qui-represente)

**BERT** c'est pour **B**idirectional **E**ncoder **R**epresentations from **T**ransformers. Il est sorti des labos [Google AI](https://ai.google/) fin 2018, et s'il est ce jour l'objet de notre attention c'est que son modèle est à la fois :

- Plus performant que ses prédécesseurs en terme de [résultats](https://rajpurkar.github.io/SQuAD-explorer/).
- Plus performant que ses prédécesseurs en terme de rapidité d'apprentissage.
- Une fois **pré-entraîné, de façon non supervisée** (initialement avec avec tout - absolument tout - le corpus anglophone de Wikipedia),  il possède une "représentation" linguistique qui lui est propre. Il est  ensuite possible, sur la base de cette représentation initiale, de le  customiser pour une tâche particulière. Il peut être **entraîné en mode incrémental (de façon supervisée cette fois)** pour spécialiser le modèle rapidement et avec peu de données.
- Enfin il peut fonctionner de façon **multi-modèle**, en prenant en entrée des  données de différents types comme des images ou/et du texte, moyennant  quelques manipulations.

Il a l'avantage par rapport à ses concurrents Open AI GTP (GPT-2 est ici pour ceux que ça intéresse) et ELMo d'être **bidirectionnel**, il n'est pas obligé de ne regarder qu'en arrière comme OpenAI GPT ou de concaténer la vue "arrière" et la vue "avant" entraînées indépendamment comme pour ELMo.

Pour faire du "sequence to sequence", i.e. de la traduction simultanée, ou du text-to-speech, ou encore du speech-to-text, l'état de l'art jusque ces dernières années, c'était les **RNNs**, nourris avec des séquences de word-embeddings , et parfois quelques couches de convolution sur les séquences d'entrée pour en extraire des caractéristiques (features) plus ou moins fines (fonction du nombre de couches), afin d'accélérer les calculs, avant de passer les infos au RNN.

Notez que **BERT utilise des embeddings sur des morceaux de mots**. Donc ni des embeddings niveau caractère, ni des embeddings au niveau de chaque mot, mais un intermédiaire.


Contrairement aux réseaux de neurones "classiques" (FFN pour feed-forward neural networks), qui connectent des couches de neurones formels les unes à la suite des autres, avec pour chaque couche sa propre matrice de poids "entraînable" - c'est à dire dont on peut modifier les poids petit à petit lors de l'apprentissage - et qui prennent en entrée des fournées (batchs) de données, **les RNNs traitent des séquences, qu'ils parcourent pas à pas, avec une même matrice de poids**. Pour cette raison (le pas-à-pas), dans le cadre des RNNs, **on ne peut pas paralléliser les calculs.**

Les séquences sont des objets dont chaque élément possède un ordre, une position, une inscription dans le temps. Par exemple dans une phrase, chaque mot vient dans un certain ordre, est prononcé sur un intervalle de temps distinct de celui des autres.


Nous allons les représenter de façon déroulée pour des raisons de lisibilité, mais en réalité il s'agit d'**itérer dans l'ordre sur chaque élément d'une séquence**. Il n'y a qu'**une seule matrice de poids, ou "les poids sont partagés dans le temps"** si vous préférez. Ci-dessous la représentation réelle et la représentation déroulée.

Une seule matrice de poids aide à traiter de **séquences de longueurs  différentes** 

2 séquences différentes peuvent avoir un  sens très similaire; dans ce cas une seule  matrice de poids permet de partager les même paramètres à chaque étape  de traitement, et par conséquence de donner un résultat global assez  similaire en sortie pour deux phrase ayant le même sens, bien qu'ayant  une composition différente.

Les FFNs sont soumis à  l'explosion ou à l'évanouissement du gradient (**exploding/vanishing  gradient descent**), et ce phénomène est d'autant plus prononcé qu'on  traite de longue séquences ou qu'on les approfondis. 

les RNNs y sont soumis quand on augmente le nombre d'étapes de  traitement, mais de façon aggravée par rapport aux FFNs. En effet il  peut y avoir des dizaines/centaines de mots dans une phrase, et **rarement autant de couches dans un FFN**. De plus, **dans un FFN, chaque couche  possède sa propre matrice de poids et ses propres fonctions  d'activation. Les matrices peuvent parfois se contre-balancer les unes  les autres et "compenser"** ce phénomène. **Dans un RNN avec une seule  matrice les problèmes de gradient sont plus prononcés.**

*Note* : Il peut y avoir plusieurs couches dans un RNN, auquel cas chaque  couche aura sa propre matrice de poids. Entendez par là qu'en sus du  nombre d'étapes de traitement, un RNN peut aussi être un réseau  "profond". On peut empiler plusieurs couches de RNN connectées entre  elles.

Afin de conserver la mémoire du contexte, et  d'atténuer les problèmes de descente de gradient, les RNNs sont  généralement composés d'unités un peu particulières, les **LSTMs** (long  short term memory), ou d'une de leur variantes les **GRUs** (gated recurrent units). Il existe en sus d'autres techniques pour les soucis de  gradient dans les RNNs - gradient clipping quand ça explose, sauts de  connexion, i.e. additive/concatenative skip/residual connexion quand ça  "vanishe" etc... 

**les LSTMs sont construites pour mémoriser une partie des autres éléments d'une séquence**, et sont donc bien adaptées à des tâches traitant d'objets **dont les  éléments ont des dépendances entre eux** à plus ou moins long terme, comme la traduction simultanée, qui ne peut pas se faire mot à mot, sans le  contexte de ce qui a été prononcé avant.

 pour faire du "sequence to sequence", la topologie utilisée  classiquement a un encodeur, suivi d'un décodeur.

* La sortie de l'encodeur est une suite de chiffres qui représente le  sens de la phrase d'entrée
* Passée au décodeur, celui-ci va générer un  mot après l'autre, jusqu'à tomber sur un élément qui signifie "fin de la phrase".

vecteur qui contient tout le sens de la phrase en sortie de  l'encodeur. Il est souvent appelé "**context vector**", vecteur contexte.



##### Les mécanismes d'attention

Les mécanismes  d'attention = les moyens de **faire passer au décodeur l'information de quelles étapes de l'encodeur (i.e. quels mots de la séquence d'entrée)  sont les plus importantes au moment de générer un mot de sortie**. Quels  sont les mots de la séquence d'entrée qui se rapportent le plus au mot  qu'il est en train de générer en sortie, soit qu'ils s'y rapportent  comme contexte pour lui donner un sens, soit qu'ils s'y rapportent comme mots "cousins" de signification proche.

**Les mécanismes d'auto-attention** (self-attention) sont similaires, sauf qu'**au lieu de s'opérer entre les éléments de l'encodeur et du décodeur**, ils **s'opèrent sur les éléments de l'input entre eux** (le présent regarde le passé et le futur) **et de l'output entre eux** aussi (le présent regarde le passé, vu que le futur est encore à générer).

Par exemple au moment de générer "feel", c'est en fait l'ensemble "avoir la pêche" qui a du sens, il doit faire attention à tous les mots, et idem  quand il génère "great", car il s'agit de la traduction d'une expression idiomatique.

**convolution**: souvent intégrée aux RNNs; répond - en partie - à un objectif similaire, fournit un "contexte" à une suite de mots. On peut en effet passer la séquence d'entrée dans une ou plusieurs couches de convolution, avant de la passer dans l'encodeur. 

Les produits de convolution vont alors extraire des caractéristiques contextuelles entre mots se trouvant à proximité les uns des autres, exacerber le poids de certains mots, et atténuer le poids de certains autres mots, et ceci de façon très positionnelle. 

**La convolution permet couche après couche d'extraire des caractéristiques (features) spatiales (dans diverses zones de la phrase), de plus en plus fines au fur et à mesure que l'on plonge plus profond dans le réseau**.

Par analogie avec la convolution sur les images, pour appliquer la convolution à une phrase on peut par exemple à mettre un mot par ligne, chaque ligne étant le vecteur d'embeddings du mot correspondant. Chaque case du tableau de chiffres que cela produit étant alors l'équivalent de l'intensité lumineuse d'un pixel pour une image.

Les CNN (convolutional neural networks) ont aussi l'avantage que les calculs **peuvent se faire en parallèle** (O(n/k) vs. O(n) pour un RNN), qu'on peut les utiliser pour concentrer l'information, et par conséquent **diminuer les problèmes de d'explosion/évanouissement du gradient**, utile s'il s'agit, par exemple, de générer une texte de plus de 1000 mots. J(e.g. bons résultats avec WaveNet et ByteNet)

Ce qui sortira du CNN sera une info du type : "les mots 8, 11 et 23 sont très importants pour donner le sens exact de cette phrase, de plus il faudra combiner ou corréler 8 et 11, retiens ça au moment de décoder la phrase". À chaque étape on décide de conserver tel ou tel mot (concentrant ainsi l'information) qui est censé avoir de l'importance au moment de générer le mot suivant. Ceci-dit, **c'est très positionnel**. Ça dépend beaucoup de la position des mots dans la phrase et de leur position les uns par rapport aux autres pour créer un contexte, plus que de leur similarité de contexte sémantique.

! différence fondamentale avec **l'attention, qui ne va pas regarder dans quelle position se trouvent les mots, mais plutôt quels sont les mots les plus "similaires" entre eux**. Comme la notion de "position" d'un élément dans une séquence reste très importante, les mécanismes d'attention nous obligeront à passer cette notion par d'autres moyens, mais ils ont **l'avantage de pouvoir faire des liens entre des éléments très distants d'une séquence, de façon plus légère qu'avec un produit de convolution,** et d'une façon plus naturelle (sans se baser sur la position mais plutôt **en comparant les similarités entre les éléments**).


Les RNNs avec LSTMs ont en réalité déjà leurs mécanismes d'attention, il existe plusieurs façons de faire; 3 grandes familles 

1. **attention stricte** (hard attention), qui **focalise sur un seul élément** du contexte  et qui est **stochastique** (pas de notion de dérivée, la sélection de  l'élément "élu" et la rétro-propagation se font via des méthodes  statistiques, échantillonnages, distributions)

- **attention  locale** qui ne sélectionne que **quelques éléments proches** sur lesquels  porter l'attention; hybride entre la soft et la hard attention.
- **attention "molle"** (soft attention), la plus classique



**Résolution de co-références - Schémas de Winograd**

lié aux mécanismes d'attention; aucun autre modèle que BERT ne donne de bonnes performances sur ce point

La **co-référence** c'est lorsque élément en référence un autre mais de  façon suffisamment ambiguë pour qu'il faille une compréhension fine de  la phrase pour comprendre ce qu'il référence.

*Exemple* : Je ne peux pas garer ma voiture sur cette place parce-qu’elle est  trop petite. <- Ici le pronom personnel "elle" renvoie à la place de  parking, pas à la voiture. Il faut une compréhension assez fine de la  phrase pour le comprendre. BERT y arrive très bien.

avec l'**attention Multi-têtes** chaque tête "voit" des choses que les autres ne voient pas et  ce "collège" se complète bien.



##### Architecture

BERT n'utilise qu'une partie de l'architecture Transformer. Comme son nom l'indique (Bidirectional Encoder Representations from Transformers) **Bert n'est composé que d'un empilement de blocs type "Encodeur" sans "Décodeur"**. Il y a aussi des modèles comme GPT-2 composés de couches "Décodeur"  seulement et plus spécialisés pour de la génération de texte. 

architecture présentée dans l'article original

- 6 **encodeurs** empilés (le Nx du schéma), 
  - chaque encodeur prenant en entrée la sortie de l'encodeur précédent (sauf le premier qui prend  en entrée les embeddings),
- suivi de 6 **décodeurs** empilés
  - prenant en  entrée la sortie du décodeur précédent et la sortie du dernier encodeur  (sauf pour le premier décodeur qui ne prend en entrée que la sortie du  dernier décodeur

Les 12 blocs (ou 24 selon les versions de BERT) **ne partagent pas les mêmes matrices de poids**.

Chaque **encodeur** 

* se compose de 2 sous-couches : 

1. une couche  d'**auto-attention "multi-têtes"** 
2. suivie d'un **FFN complètement connecté et position-wise** (i.e. chaque élément du vecteur de sortie de la couche  précédente est connecté à un neurone formel de l'entrée du FFN, dans le  même ordre qu'ils le sont dans le vecteur). 

* Chaque sous-couche possède  en sortie **une couche qui ajoute, additionne, les sorties de la couche et du raccord à une connexion dite résiduelle** (qui connecte directement  les valeurs d'entrée de la couche à la sortie de la couche) et qui  **normalise** l'ensemble.

Chaque **décodeur** 

* se compose de 3 couches : 
  1. une couche d'**auto-attention "multi-têtes",** 
  2. suivie d'une couche d'**attention avec le dernier encodeur**, 
  3. puis un **FFN complètement  connecté et position-wise** (i.e. chaque élément du vecteur de sortie de  la couche précédente est connecté à un neurone formel de l'entrée du  FFN, dans le même ordre qu'ils le sont dans le vecteur). 
* Chaque  sous-couche possède **en sortie une couche qui ajoute, additionne, les  sorties de la couche et du raccord à une connexion dite résiduelle** (qui  connecte directement les valeurs d'entrée de la couche à la sortie de la couche) et qui **normalise** l'ensemble.



**3 mécanismes  d'attention type "clé-valeur"** 

1. **Auto-attention** (dans l'encodeur)
2. **Auto-attention avec les tous  éléments précédemment générée** (en entrée du décodeur)
3. **Attention "masquée"** (dans le décodeur) (masked attention, parce-qu’on applique un masque) nous verrons les détails plus loin dans l'article) entre  **l'élément à générer dans le décodeur et tous les éléments de l'encodeur**. 

Les couches d'attention ont plusieurs têtes

Pour la génération de texte ou la traduction simultanée le **mécanisme est auto-régressif**, c'est à dire qu'on fait entrer une séquence dans le premier encodeur,  la sortie prédit un élément, puis on repasse toute la séquence dans  l'encodeur et l'élément prédit dans le décodeur en parallèle afin de  générée un deuxième élément, puis à nouveau la séquence dans l'encodeur  et tous les éléments déjà prédits dans le décodeur en parallèle etc...  jusqu'à prédire en sortie un <fin de séquence>.

En sortie une distribution de probabilité qui permet de prédire l'élément de sortie le plus probable.

Et le tout **se passe complètement de LSTMs !** 



##### Comment BERT apprend

de façon non supervisée, l'entrée se suffit à elle même, pas besoin de labelliser

**Masked language model**

[CLS] indique un début de séquence

[SEP] une séparation, en général entre deux phrases dans notre cas.

[MASK] un mot masqué

<u>Les mots "masqués"</u>


Ici la séquence d'entrée a été volontairement oblitérée d'un mot, le mot masqué, et **le modèle va apprendre à prédire ce mot masqué.**

<u>La phrase suivante</u>


Ici le modèle doit **déterminer si la séquence suivante (suivant la séparation[SEP]) est bien la séquence suivante**. Si oui, IsNext sera vrai, le label sera IsNext, si non, le label sera NotNext.



##### 	Les customisations possibles

une fois son  apprentissage non supervisé terminé, il est :

* capable de se spécialiser sur beaucoup de tâches différentes  (traduction, réponse à des questions etc.)

*  surclasse dans la plus part de ses spécialisations les modèles spécialisés existants



##### 	Les sous-parties du Transformer en détails

exemple pour la génération de texte

<u>en entrée</u>

* embeddings : chaque **mot** est représenté par un vecteur (colonne ou ligne de réels)
* On ajoute éventuellement à ces embeddings, pour chaque mot, les embeddings d'un "**segment**" quand cela a du sens (par exemple chaque phrase est un  segment et on veut passer plusieurs phrases à la fois, on va alors dire  dans quel segment se trouve chaque mot).
* On ajoute  ensuite le "**positional encoding**", qui est une façon d'encoder la place  de chaque élément **dans la séquence**. 
  * Comme la longueur des phrases n'est  pas prédéterminée, on va utiliser des fonctions sinusoïdales donnant de  petites valeurs entre 0 et 1, pour modifier légèrement les embeddings de chaque mot. 
  * La dimension de l'embedding de position (à sommer avec  l'embedding sémantique du mot) est la même que celle de l'embedding  sémantique, soit 512, pour pouvoir sommer terme à terme.
  * il existe beaucoup de façons d'encoder la position d'un élément dans une séquence.



Ce qui donne en entrée une matrice de taille [longueur de la séquence] x [dimension des embeddings - 512] 



##### 		L'attention dans le cas spécifique du Transformer



mécanisme d'attention (auto-attention ou pas) de **type clé-valeur**.



Chaque mot, décrit comme la **somme de ses embeddings sémantiques et positionnels** va être décomposé en **trois abstractions** :

1. Q = Une **requête** (query)
2. K = Une **clé** (key)
3. V = Une **valeur** (value)

Dans l'exemple, chacune de ces abstractions est ici un vecteur de dimension 64. Comme on veut des sorties de dimension [longueur de la séquence] x [dimension des embeddings - 512] tout au long du  parcours, et que dans l'attention on fera du multi-têtes (voir plus  loin) avec 8 têtes, qu'on concaténera la sortie de chaque tête, on aura  alors 8 x 64 = 512 en sortie de l'attention et c'est bien ce qu'on veut.

**Chacune de ces abstractions est calculée et apprise** (mise à jour des matrices  de poids) lors du processus d'apprentissage, grâce à une matrice de  poids. Chaque matrice est distincte.

Les dimensions de ces matrices sont [64 (dimension requête ou clé ou valeur)] x [longueur de la séquence].



**Multi-têtes** :

*  **Chaque tête de l'attention a ses propres matrices de poids** 
* on **concatène la sortie de chaque tête** pour retrouver une matrice de  dimension [longueur de la séquence] x [dimension des embeddings, i.e.  512].

explication de la formule de l'attention

* [Q x Transposée de K] est un produit scalaire entre les vecteurs  requête et les vecteurs clé.
  * plus la clé  "ressemblera" à la requête, plus le score produit par [Q x Transposée de K] sera grand pour cette clé.
* La partie *dk (=64)* pour normaliser, pas toujours utilisé dans les mécanismes d'attention.
* softmax donne une distribution de probabilités qui va encore  augmenter la valeur pour les clés similaires aux requêtes, et diminuer  celles des clés dissemblables aux requêtes.
* les clés  correspondent à des valeurs, quand on multiplie le résultat précédent  par V, **les valeurs correspondant aux clés qui ont été "élues" à l'étape  précédente sont sur-pondérées par rapport aux autres valeurs.**

Enfin on concatène la sortie de chaque tête, et on multiplie par une matrice W0, de dimensions 512 x 512 ([(nombre de têtes) x (dimension requête ou clé ou valeur, i.e. 64)]x[dimension des embeddings]), qui apprend à projeter le résultat sur un espace de sortie aux dimensions attendues.

##### 		La connexion résiduelle

**ajouter la représentation initiale** à celle calculée dans les couches d'attention ou dans le FFN.

-> *Cela revient à dire* : Apprend les relations entre les éléments de la séquence, mais n'oublie pas ce que tu sais déjà à propos de toi.

appliquer un dropout de 10%, donc en sortie de chaque couche i



##### Le FFN

Couches de neurones formels avec une ReLU comme fonction d'activation

Ici W1 a pour dimensions [dimension des embeddings]x[dimmension d'entrée du FFN - au choix] et W2 [dimmension d'entrée du FFN - au choix] x  [dimension des embeddings

C'est le réseau de neurones "standard" 



##### *Rappelons le principe d'auto-régression du Transformer*.

Par exemple pour traduire "Avoir la pèche", nous avons vu que :

* 1er passage, on envoie 
  * dans l'encodeur une version  "embeddée" de la séquence [<CLS> Avoir la pèche <fin de  phrase>], 
  * en même temps l'amorce de séquence [<CLS>] dans le décodeur.
  * Le Transformer doit alors prédire le 1er  mot de la traduction : "feel" (i.e. nous sortir la séquence [<CLS> feel])
* 2ème passage, on envoie 
  * dans  l'encodeur de nouveau toute la version "embeddée" de la séquence [<CLS> Avoir  la pèche <fin de phrase>],
  * une version "embeddée" de ce que le  Transformer a prédit dans le décodeur.

*  etc... jusqu'à ce que la séquence prédite en sortie du Transformer finisse par <fin de phrase>.



##### L'auto-attention "masquée" :

​	*- Pourquoi :*

Dans notre cas, la phrase a prédire en sortie est déjà connue lorsque nous entraînons le modèle. Le jeu  d'entraînement possède déjà la correspondance entre "avoir la pèche" et  "feel great"

Or il faut faire apprendre au modèle que :

*  Au premier passage quand l'encodeur reçoit  [<CLS> Avoir la pèche  <fin de phrase>] et le décodeur reçoit  [<CLS>], le modèle  doit prédire  [<CLS> feel].
* Puis "feel" dans ce contexte doit prédire "great" au passage suivant.
* Etc...

Il faut donc lors de l'entraînement **"masquer" à l'encodeur le reste des  mots à traduire**. Quand "feel" sort du modèle, le décodeur ne doit pas  voir "great", **il doit apprendre seul que "feel" dans ce contexte doit  ensuite donner "great"**.

​	*- Comment :*

Eh bien on va simplement faire en sorte que dans la matrice softmax([Q x Transposée de K]) de notre formule  d'attention, la ligne correspondant a chaque mot soit a **0 pour les  colonnes représentant les mots suivants "chronologiquement" le dernier  mot généré par le modèle**.

on veut obtenir de quoi masquer à chaque mot prévu les mots qu'il devrait prévoir

donc on veut que les valeurs correspondantes aux mots à venir n'aient aucune attention

il faut donc que ce soit multipliée avec V une matrice dont les valeurs dans le triangle supérieur = 0; pour ce faire on force QK^T à avoir les valeurs au-dessus de la diagonale de la matrice à moins l'infini



##### L'attention encodeur-décodeur 

Le décodeur possède une couche d'attention qui 

* prend en entrée le séquence de sortie du FFN de l'encodeur, 
* qu'il multiplie à ses matrices clés et  valeurs (Wki et Wvi pour la tête "i"), 
* tandis que la séquence sortant de la couche d'auto-attention "masquée" de l'encodeur va se multiplier à  la matrice des requêtes (Wqi pour la tête "i").

L'encodeur  découvre 

* des choses intéressantes (features) à propos de la séquence  d'entrée => les **valeurs**. 
* À ces valeurs attribue un label, un  index, une façon d'adresser ces "choses" => la **clé**. 

Puis le décodeur avec sa **requête** va décider du type de valeur à aller chercher. **La  requête demande la valeur pour une clé en particulier.**





##### La sortie



*La couche linéaire*

Enfin nous y voila. Appelons S la matrice en sortie du décodeur. On la  multiplie par une matrice de poids (qui peuvent apprendre) W1. C'est une **couche totalement connectée** qui projette simplement la sortie précédente dans un espace de la taille de notre vocable.

W1 est la matrice qui va permettre d'extraire un mot dans notre  dictionnaire de vocabulaire. Elle aura donc pour dimensions [dimension  des embeddings, i.e. *dmodel*] x [nombre de mots dans notre vocable].



*La softmax*

c'est sûrement la dernière ligne de S.W1  (correspondant au dernier mot généré, de dimension [1] x [taille du  vocable]) qui est passée par la softmax. 

La softmax  nous donne alors l'élément le plus probable à prédire (on prend le mot  de la colonne qui donne la probabilité la plus haute).



### [Dissecting BERT Part 1: The Encoder](https://medium.com/dissecting-bert/dissecting-bert-part-1-d3c3d495cdb3)

The **Encoder** used in BERT is an **attention-based architecture** for Natural Language Processing (NLP) 

the Transformer is composed of two parts, the Encoder and the Decoder. 

**BERT only uses the Encoder** 

##### information flow

1. each token represented as a vector of *emb_dim* size. 1 **embedding vector** for each of the input tokens -> *(input_length) x (emb_dim)* matrix for a specific input sequence.
2. then adds positional information (**positional encoding**). This step returns a matrix of dimensions *(input_length) x (emb_dim)*, just like in the previous step.
3. The data goes through N **encoder blocks**. After this, we obtain a matrix of dimensions *(input_length) x (emb_dim)*.

dimensions of the input and output of the encoder block are the same -> makes sense to use the output of one encoder block as the  input of the next encoder block

the number of blocks N was chosen to be 12 and 24.

the blocks do not share weights with each other

##### Tokenization, numericalization and word embeddings

The first step is to **tokenize** it:

followed by **numericalization**, mapping each token to a unique integer in the corpus’ vocabulary.

 get the **embedding** for each word in the sequence. Each word of the sequence is mapped to a ***emb_dim* dimensional vector that the model will learn during training**. You can think about  it as a vector look-up for each token. The elements of those vectors are **treated as model parameters** and are optimized with back-propagation  just like any other weights.

padding was used to make the input sequences in a batch have the same length. That is, we increase the length of some of the sequences by adding ‘<pad>’ tokens

##### Positional Encoding

At this point, we have a matrix representation of our sequence. However, these representations are not encoding the fact that words appear in  different positions

 we aim to be able to **modify the represented meaning of a specific word  depending on its position**. We don't want to change the full  representation of the word but **we want to modify it a little to encode  its position**.

The approach chosen in the paper is to **add numbers between *[-1,1]* using predetermined (non-learned) sinusoidal functions to the token embeddings**. 

 the word will be represented slightly differently depending on the position the word is in (even if it is the same word).

Moreover, we would like the Encoder to be able to use the fact that some words are in a given position while, in the same sequence, other words are in other specific positions -> the network to be able to **understand relative positions and not only absolute ones**. 

**sinuosidal functions allow positions to be represented as linear combinations of each other and thus allow the network to learn relative relationships between the token positions.**

The approach chosen in the paper to add this information is adding to Z a matrix P with positional encodings.

The authors chose to use a combination of sinusoidal functions. Mathematically, using i for the position of the token in the sequence and j for the position of the embedding feature

advantages over learned positional representations:

* The input_length can be increased indefinitely since the functions can be calculated for any arbitrary position.
* Fewer parameters needed to be learned and the model trained quicker.

The resulting matrix is the input of the first encoder block and has dimensions (input_length) x (emb_dim).

##### Encoder block

N encoder blocks are chained together to generate the Encoder’s output

A specific block is in charge of finding relationships between the input representations and encode them in its output.

Intuitively, this **iterative process** through the blocks will help the neural network  **capture more complex relationships between words** in the input sequence.  You can think about it as **iteratively building the meaning of the input  sequence as a whole**.



##### Multi-Head Attention

= it computes attention h different times with different weight matrices and then concatenates the results together.

The result of each of those parallel computations of attention is called a **head**. 

once all the heads have been computed they will be concatenated

This will result in a matrix of dimensions *(input_length) x (h*d_v). Afterwards, a linear layer with weight matrix W⁰ of dimensions (h*d_v) x (emb_dim) will be applied leading to a final result of dimensions (input_length) x (emb_dim). 

##### Scaled Dot-Product Attention

Each head is going to be characterized by 3 different projections (matrix multiplications) 

To compute a head we will take the input matrix X and separately project it with the above weight matrices

Once we have K_i, Q_i and V_i we use them to compute the Scaled Dot-Product Attention



**In the encoder block the computation of attention does not use a mask.**

This is the key of the architecture (the name of the paper is no  coincidence) so we need to understand it carefully. Let’s start by  looking at the matrix product between *Q_i* and *K_i* transposed:

![img](https://miro.medium.com/max/60/1*szTtSJSZBfej5q-KpLmf3Q.png?q=20)



*Q_i* and *K_i* = different projections of the tokens into a *d_k* dimensional space. 

we can think about the **dot product** of those projections as a **measure of similarity between tokens projections**

For every vector projected through *Q_i* the dot product with the projections through *K_i* measures the similarity between those vectors. 

 this is a measure of how similar are the directions of u_i and v_j and how large are their lengths (the closest the direction and the larger the length, the greater the dot product).

Another way of thinking about this matrix product is as the **encoding of a specific relationship between each of the tokens in the input sequence** (the relationship is defined by the matrices K_i, Q_i).

After this multiplication, the matrix is divided element-wise by the square root of d_k for **scaling** purposes.

The next step is a **Softmax** applied row-wise (one softmax computation for each row)

Thus, at this point, the representation of the token is the concatenation of *h* weighted combinations of token representations (centroids) through the *h* different learned projections.

##### Position-wise Feed-Forward Network

3 layers: FC linear layer -> ReLU -> FC linear layer

during this step, **vector representations of tokens don’t “interact” with each other.** It is equivalent to run the calculations row-wise and stack the resulting rows in a matrix

The output of this step has dimension *(input_length) x (emb_dim)*.

##### Dropout, Add & Norm

Before this layer, there is always a layer for which inputs and outputs have the same dimensions (Multi-Head Attention or Feed-Forward). We will call that layer Sublayer and its input x.

After each Sublayer, **dropout** is applied with 10% probability. Call this result Dropout(Sublayer(x)). This result is added to the Sublayer’s input x, and we get x + Dropout(Sublayer(x)).

Observe that in the context of a Multi-Head Attention layer, this means **adding the original representation** of a token x to the representation based on the relationship with other tokens. It is like telling the token:

“Learn the relationship with the rest of the tokens, but don’t forget what we already learned about yourself!”

Finally, a **token-wise/row-wise normalization** is computed with the mean and standard deviation of each row. This **improves the stability** of the network.

### [Dissecting BERT Appendix: The Decoder](https://medium.com/dissecting-bert/dissecting-bert-appendix-the-decoder-3b86f66b0e5f)





### [French NLP: entamez le CamemBERT avec les librairies fast-bert et transformers](https://medium.com/@vitalshchutski/french-nlp-entamez-le-camembert-avec-les-librairies-fast-bert-et-transformers-14e65f84c148)

Afin de révéler le potentiel de CamemBERT il faudra adapter son modèle de langage à nos données. En anglais on parle de fine-tuning. Si vous êtes novice dans le NLP, on peut comparer cette opération à l’ajustement d’un costume trop large à votre taille. Attention, **on ne pré-entraîne pas CamemBERT** à nouveau. Ce modèle “maîtrise déjà la grammaire de la langue française”. On l’aide simplement à mieux comprendre la langue et la structure des commentaires. Lors du fine-tuning il faudra utiliser un learning rate très bas, car il s’agit des petits ajustements et non pas de ré-apprentissage.





Les deux principales méthodes d'apprentissage automatique de Word2Vec sont Skip-gram et Continuous Bag of Words.

* Le modèle **Skip-gram** prédit les mots (contexte) autour du mot cible (cible) (cible -> contexte)
* le modèle **Continuous Bag of Words** prédit le mot cible à partir des mots autour de la cible (contexte) (contexte -> cible)

Le mot cible ne doit pas nécessairement se trouver au centre de la «**fenêtre contextuelle**» qui est composée d'un nombre donné de mots environnants, mais peut se trouver à gauche ou à droite de la fenêtre contextuelle.

Un point important à noter est que **les fenêtres contextuelles mobiles sont unidirectionnelles**. C'est-à-dire que la fenêtre se déplace sur les mots dans une seule direction, de gauche à droite ou de droite à gauche.

en plus de ceux de son nom, BERT apporte d'autres développements passionnants dans le domaine de la compréhension du langage naturel.

* Pré-formation à partir d'un texte non étiqueté
* Modèles contextuels bidirectionnels
* L'utilisation d'une architecture de transformateur
* Modélisation du langage masqué
* Attention focalisée
* Implication textuelle (prédiction de la phrase suivante)
* Désambiguïsation grâce au contexte open source

La  magie  du BERT est sa mise en œuvre d'une **formation bidirectionnelle sur un corpus de texte non étiqueté**, 



BERT a été **le premier framework / architecture de langage naturel à être pré-formé en utilisant un apprentissage non supervisé sur du texte brut** pur (2,5 milliards de mots + de Wikipedia anglais) plutôt que sur des corpus étiquetés.

Les anciens modèles de formation en langage naturel ont été formés de manière **unidirectionnelle**. La signification du mot dans une fenêtre contextuelle s'est déplacée de gauche à droite ou de droite à gauche avec un nombre donné de mots autour du mot cible (le contexte du mot ou «c'est la société»). Cela signifie que **les mots qui ne sont pas encore vus dans leur contexte ne peuvent pas être pris en considération** dans une phrase et qu'ils pourraient en fait changer le sens d'autres mots en langage naturel. Les fenêtres contextuelles mobiles unidirectionnelles peuvent donc manquer certains contextes changeants importants.

attention - Essentiellement, le BERT est capable de regarder tout le contexte dans la cohésion du texte en concentrant l'attention sur un mot donné dans une phrase tout en identifiant également tout le contexte des autres mots par rapport au mot. Ceci est réalisé simultanément en utilisant des transformateurs combinés avec une pré-formation bidirectionnelle.

Cela contribue à un certain nombre de défis linguistiques de longue date pour la compréhension du langage naturel, y compris la résolution de coréférence. Cela est dû au fait que les entités peuvent être ciblées dans une phrase en tant que mot cible et que leurs pronoms ou les phrases nominales les référençant sont résolus de nouveau vers l'entité ou les entités dans la phrase ou l'expression.

De plus, l'attention focalisée aide également à la désambiguïsation des mots polysémiques et des homonymes en utilisant **une prédiction / pondération de probabilité basée sur le contexte entier du mot en contexte avec tous les autres mots de la phrase**.  Les autres mots reçoivent un score d'attention pondéré pour indiquer  combien chacun ajoute au contexte du mot cible en tant que  représentation du «sens».

 L'encodeur est l'entrée de phrase traduite en représentations de sens  des mots et le décodeur est la sortie de texte traitée sous une forme  contextualisée.



**Modélisation du langage masqué** (formation MLM)

Également connue sous le nom de «**procédure Cloze**» qui existe depuis très longtemps. L'architecture BERT analyse les phrases avec certains mots masqués de manière aléatoire et tente de prédire correctement ce qu'est le mot «caché».

Le but de ceci est **d'empêcher les mots cibles dans le processus d'apprentissage passant par l'architecture du transformateur BERT de se voir par inadvertance pendant l'entraînement bidirectionnel** lorsque tous les mots sont examinés ensemble pour un contexte combiné. C'est à dire. cela évite un type de boucle infinie erronée dans l'apprentissage automatique du langage naturel, qui fausserait le sens du mot.



Implication textuelle (**prédiction de la phrase suivante**)

L'une des principales innovations du BERT est qu'il est censé être capable de prédire ce que vous allez dire ensuite

formé pour prédire à partir de paires de phrases si la deuxième phrase fournie correspond bien à un corpus de texte.

L'implication textuelle est un type de "qu'est-ce qui vient ensuite?" dans un corps de texte. En plus de l'implication textuelle, le concept est également connu sous le nom de «prédiction de la phrase suivante».

 L'implication textuelle est une tâche de traitement du langage naturel impliquant des paires de phrases. 

La première phrase est analysée, puis un niveau de confiance déterminé pour prédire si une deuxième phrase hypothétique donnée dans la paire «correspond» logiquement à la phrase suivante appropriée, ou non, avec une prédiction positive, négative ou neutre, à partir d'un texte collection sous examen.



    