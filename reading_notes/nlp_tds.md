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

the Decoder in depth; the part of the Transformer architecture that are not used in BERT. 

The problem that the Transformer addresses is translation. To translate a sentence into another language, we want our model to:

* Be able to capture the relationships between the words in the input sentence.
* Combine the information contained in the input sentence and what has already been translated at each step.

Imagine that the goal is to translate a sentence from English to Spanish 

* First, we want to process the information in the input sequence X by **combining the information in each of the words of the sequence** -> done inside the **Encoder**.
* Once we have this information in the output of the Encoder we want to **combine it with the target sequence** -> done in the **Decoder**.

Encoder and Decoder are specific parts of the **Transformer** architecture

**Information Flow**

The data flow through the architecture is as follows:

* (1) represents **each token as a vector** of dimension *emb_dim* -> matrix of dimensions *(input_length) x (emb_dimb)* for a specific input sequence
* (2) adds positional information (**positional encoding**) -> matrix of dimensions *(input_length) x (emb_dim)*
* (3) data goes through N **encoder blocks** -> matrix of dimensions *(input_length) x (emb_dim)*
* (4) target sequence **masked** and sent **through the decoder’s** **equivalent of 1) and 2)** -> *(target_length) x (emb_dim)* output
* (5) result of 4) goes through N **decoder blocks**. In each of the iterations, the decoder is using the encoder’s output 3) ->  (target_length) x (emb_dim) output
* (6) applies a **fully connected layer** and a **row-wise softmax** -> *(target_length) x (vocab_size)* output

the described algorithm is processing both the input sentence and the target sentence to train the network

input sentence encoded in The Encoder’s architecture

in the decoder: how **given a target sentence we obtain a matrix representing the target sentence** for the decoder blocks

same process, composed of two general steps:

* Token embeddings
* Encoding of the positions.

main difference: **the target sentence is shifted** -> before padding, the target sequence will be as follows:

The rest of the process to vectorize the target sequence = as the one described for input sentences in The Encoder’s architecture.

##### Decoder block — Training vs Testing

During **test time** we don’t have the ground truth. The steps, in this case, will be as follows:

1. Compute the **embedding representation of the input** sequence.
2. Use a **starting sequence token**, for example ‘<SS>’ **as the first target sequence**: [<SS>] -> output = the next token.
3. **Add the last predicted token to the target sequence** and **use it to generate a new prediction** [‘<SS>’, Prediction_1,…,Prediction_n]
4. Do step 3 **until the predicted token is the one representing the End of the Sequence**, for example <EOS>.

During **training** we have the ground truth, i.e. the tokens we would like the model to output for every iteration of the above process. Since **we have the target in advance**, we will **give the model the whole shifted target sequence at once and ask it to predict the non-shifted target**.

However, there is a problem here. What **if the model sees the expected token and uses it to predict itself**? For example, it might see ‘estas’ at the right of ‘como’ and use it to predict ‘estas’. That’s **not what we want because the model will not be able to do that a testing time**.

 modify some of the attention layers to **prevent the model of seeing information on the right** (or down in the matrix of vector representation) **but allow it to use the already predicted words**.

transform the matrix of representation and add positional encoding

as in the encoder the output of the decoder block will be also a matrix of sizes *(target_length) x (emb_dim).* 

After a **row-wise linear layer** (a linear layer in the form of matrix product on the right) and a **Softmax** per row this will result in a matrix for which **the maximum element per row indicates the next word**.

we don’t have problems in the linear layers because they are defined to be token-wise/row-wise in the form of a matrix multiplication through the right

The problem will be in Multi-Head Attention and the input will need to be masked

At training time, the prediction of all rows matter. Given that at prediction time we are doing an iterative process we are just going to care about the prediction of the next word of the last token in the target/output sequence.

##### Masked Multi-Head Attention

This will work exactly as the Multi-Head Attention mechanism but **adding masking to our input**.

T**he only Multi-Head Attention block where masking is required is the first one of each decoder block.** 

the one in the middle is used to combine information between the encoded inputs and the outputs inherited from the previous layers. There is no problem in combining every target token’s representation with any of the input token’s representations (since we will have all of them at test time).

The modification will take place after computing the QK/sqrt(d) ratio matrix

the masking step is just going to set to minus infinity all the entries in the strictly upper triangular part of the matrix

* if those entries are relative attention measures per each row, the larger they are, the more attention we need to pay to that token.
* softmax output: the relative attention of those tokens that we were trying to ignore has indeed gone to zero.
* When multiplying this matrix with V_i the only elements that will be accounted for to predict the next word are the ones into its right, i.e. the ones that the model will have access to during test time.

the output of the modified Multi-Head Attention layer will be a matrix *(target_length) x (emb_dim)* because the sequence from which it has been calculated has a sequence length of target_length.

The rest of the process is identical as described in the Multi-Head Attention for the encoder.

##### Multi-Head Attention — Encoder output and target

Observe that in this case we are using different inputs for that layer. More specifically, instead of deriving Q_i, K_i and V_i from X as we have been doing in previous Multi-Head Attention layers, this layer will use both the Encoder’s final output E (final result of all encoder blocks) and the Decoder’s previous layer output D (the masked Multi-Head Attention after going through the Dropout, Add & Norm layer).

[...]

**every token in the target sequence is represented in every head as a combination of encoded input tokens**. Moreover, this will happen for multiple heads and just as before, that is going to **allow each token of the target sequence to be represented by multiple relationships with the tokens in the input sequence**.



##### Linear and Softmax

This is the final step before being able to get the predicted token for every position in the target sequence. 

 The output from the last Add & Norm layer of the last Decoder block is a matrix X *(target_length)x(emb_dim)*.

linear layer: for every row in x of X compute xW_1

where W_1 is a matrix of learned weights of dimensions *(emb_dim) x (vocab_size)* -> for a specific row the result will be a vector of length *vocab_size*.

a **softmax** is applied to this vector -> **vector describing the probability of the next token**. Therefore, **taking the position corresponding to the maximum probability returns the most likely next word according to the model**.

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



### [Apprentissage de Représentation dans les Réseaux de Documents : Application à la Littérature Scientifique](https://tel.archives-ouvertes.fr/tel-02899422/document)



Le succès des algorithmes d’apprentissage artificiel dépend principalement des repré-
sentations des données sur lesquelles ils sont appliqués.

**L’apprentissage de représentation** (AR) [Bengio et al., 2013] s’oppose à la construction manuelle de caractéristiques des
données en apprenant automatiquement des descriptions qui rendent leur analyse plus
efficace. En d’autres termes, plutôt que de construire à la main des représentations des
données en utilisant des connaissances expertes, l’AR désigne une approche différente où
l’on va construire ces représentations par un algorithme d’apprentissage optimisant un ou
plusieurs critères sur les données.

Un critère couramment utilisé en apprentissage de représentation du texte s’appuie sur
**l’hypothèse distributionnelle**. Celle-ci stipule que **des mots apparaissant dans les mêmes**
**contextes linguistiques partagent des significations similaire**



 partant d’une re-
présentation symbolique du texte (séquences de jetons) et transcrivant cette hypothèse
dans un espace euclidien muni d’une mesure de similarité, nous sommes en mesure de
construire des vecteurs denses associés aux mots du vocabulaire en rapprochant les mots
apparaissant dans les mêmes contextes linguistiques.

Cette méthodologie est à l’origine
des méthodes de plongement de mot (**word embedding**) [Mikolov et al., 2013a] qui a en-
suite été étendue au plongement de réseau

Les techniques d’apprentissage de représentation pour le texte et pour les réseaux sont
intimement liés. 

En effet, ces deux types de données sont naturellement représentées par
des ensembles fini d’éléments (ex : mots et sommets) dont on peut mesurer des simila-
rités deux à deux (ex : nombre de co-occurrences entre mots et nombre de marches où
apparaissent des sommets).

es mêmes méthodes de repré-
sentations, tels que les plongements ou les mécanismes d’attention, soient appliquées à
ces deux types de données. 

Un cas simple et largement utilisé d’algorithme de **plongement de réseau** est l**’algo-**
**rithme de Fruchterman-Reingold** [Fruchterman and Reingold, 1991]. Celui-ci permet de
représenter les sommets dans un espace vectoriel à deux dimensions, rendant ainsi la vi-
sualisation d’un réseau plus digeste; C’est une méthode itérative qui simule, comme un
modèle physique, une attraction des sommets connectés et une répulsion latente qui vise
à séparer toute paire de sommets.

Originellement, le plongement de graphe est utilisé comme une méthode générale de
réduction de dimension

L’émergence des algorithmes de plongement de réseau a suivi celle des algorithmes
de plongement de mot (word embedding).

Les méthodes d’apprentissage de représentation pour le texte et pour les réseaux sont
fortement liées. En effet, ces deux types de données peuvent être décrits par des éléments
symboliques (par exemple les mots et les sommets) dont les relations sont mesurables
(par exemple en terme de co-occurrences des mots dans un corpus et de sommets dans
des marches aléatoires).

L’**hypothèse distributionnelle** est une hypothèse fondamentale des algorithmes de plon-
gement de mots. Celle-ci stipule que **la similarité distributionnelle des mots est fortement**
**corrélée avec la similarité de leurs sens**.

*  pour construire des repré-
  sentations vectorielles captant le sens des mots, il suffit d’étudier le voisinage de ceux-ci,
  c’est-à-dire le contexte dans lequel ils apparaissent.
*  si un modèle est
  capable de reconstruire les mots contextes d’un certain mot cible, il est capable d’en
  représenter le sens

**Skip-Gram** [Mikolov et al., 2013a], l’une des variantes de la suite logicielle Word2vec est un modèle qui **construit deux représentations pour chaque mot** ωi : **un vecteur cible**
ui et **un vecteur contexte** hi. Ces deux vecteurs sont utilisés pour **calculer la probabilité**
**conditionnelle d’observer un mot selon son contexte**, exprimée comme la fonction **softmax
du produit scalaire de leurs représentations** 

 Cet ensemble est construit en faisant glisser une fenêtre de taille τ sur un corpus de
texte

 Skip-Gram modélise les probabilités d’occurrence d’un mot cible conditionnellement
à chaque mot contexte, de manière indépendante

a popularité de Skip-Gram vient aussi des propriétés géométriques des repré-
sentations qu’il construit. En effet, celles-ci semblent être en lien direct avec les sens
sémantiques et syntaxiques des mots.

DeepWalk [Perozzi et al., 2014] est une méthode de plongement de réseau qui s’inspire
de Skip-Gram. 

’intuition centrale de cette approche est que les chemins générés par de
courtes marches aléatoires dans un graphe sont similaires à des phrases en langage naturel.

La fréquence d’apparition des sommets dans ces marches suit une loi puissance, similai-
rement aux fréquences des mots dans un corpus

l’exemple de la traduction
automatique, particulièrement telle qu’elle est abordée dans un modèle de type encodeur-
décodeur neuronal [Bahdanau et al., 2014]. La tâche consiste à produire en sortie une
séquence de plongements de mots (y1,...,y_y) étant donnée une séquence de plongements
de mots d’entrée (x1,...,x_x).

Les mots en sortie correspondent par exemple à de l’anglais
alors que les mots en entrée correspondent à du français.

Le processus est **auto-régressif**, ce
qui signifie que l**a séquence de sortie est générée mot par mot**.

À chaque étape i, l’encodeur
utilise la séquence d’entrée (x1,...,x`x) et le morceau de séquence de sortie précédemment
généré (y1,...,yi−1) afin de prédire le mot suivant yi.

Notons que l’on définit le premier
vecteur de sortie comme une constante y1 = ystart ce qui permet de définir la récurrence
pour la première itération. 

le problème auquel cette approche fait face est la gestion de
la diversité d’information présente en entrée pour prédire chaque sortie

 En traduction
automatique, prédire un mot ne dépend souvent que d’une infime proportion des vecteurs
présentés à l’encodeur-décodeur (x1,...,x`x,y1,...,yi−1). 

. Lorsque cette séquence est longue,
il devient difficile pour le modèle d’en faire le tri. C’est précisément pour faciliter ce tri que
les **mécanismes d’attention** sont utilisés. Ils vont **permettre au modèle de ne sélectionner qu’un sous-ensemble précis de la séquence d’entrée afin de prédire le prochain mot**.

Le premier mécanisme d’attention pour les réseaux profonds est présenté dans [Xu
et al., 2015] pour la génération automatique de légendes pour images. Le modèle est dé-
crit comme un **encodeur-décodeur**. L’encodeur extrait un ensemble de N représentations
vectorielles (a1,...,aN) d’une image via un **CNN**, appelées vecteurs d’annotation. Ces vec-
teurs sont générés de sorte à capturer les différentes composantes de l’image. Le décodeur,
un **LSTM**, produit de manière auto-régressive une séquence de mots en sortie. 

Pour ce
faire, il prend en entrée les mots précédemment générés (y1,...,yi−1). **L’attention est in-**
**tégrée pour conditionner cette génération au vecteurs d’annotation**. 

Le **vecteur contexte**
ci du LSTM est calculé non plus comme une fonction de ci−1 et de hi−1 mais comme une
**moyenne pondérée des vecteurs d’annotation issus de l’image** 

Les poids
αk sont calculés par produit scalaire entre le vecteur caché hi−1 de la précédente étape
(capturant l’historique des mots générés (y1,...,yi−1)) avec chacun des vecteurs d’annota-
tion ak.

chaque mot est généré en confrontant les caractéristiques de l’image
et l’historique des mots précédemment générés en légende

 Ceci permet au modèle de se
concentrer alternativement sur les différents éléments de l’image afin de construire une
description en langage naturelle de celle-ci

en étudiant les poids αk, on peut
facilement identifier la région de l’image ayant motivé le choix du mot généré par le
modèle

Le **Transformer** [Vaswani et al., 2017] est **le premier modèle de traduction automa-**
**tique reposant uniquement sur un mécanisme d’attention, sans RNN ni CNN**

**L’aspect**
**séquentiel de l’entraînement d’un RNN (prédire chaque mot l’un après l’autre) consti-**
**tue le véritable point faible** de ces architectures car il rend difficile leur parallélisation.

Le Transformer reprend l’architecture **encodeur-décodeur** couramment utilisée en traduc-
tion automatique et introduit un **mécanisme d’attention**, le **scaled dot-product attention**
(SDPA), **parallélisable** et reposant principalement sur des opérations matricielles.

Dans
sa version la plus simple, SDPA transforme une séquence de vecteurs d’entrée (x1,...,x_l)
en une séquence de vecteurs de sortie (y1,...,y_l) dont **chaque représentation yi repré-*
sente xi conditionnellement à l’ensemble des vecteurs (x1,...,x_l).** 

SDPA construit 3 représentations distinctes des xi par projections linéaires : 

* les **requêtes** Q = XWq,
* les **clefs** K = XWk et 
* les **valeurs** V = XWv. 

En notant ρw la dimension de toutes les représentations X,Y,Q,K et V , le mécanisme d’attention s’écrit :
$ Y = softmax( \frac{QK^T}{√ρ_w})V$ 

le **nominateur** = une matrice dont chaque valeur contient le produit scalaire entre une requête qi et une clef kj. 

le **dénominateur** permet de réduire le problème de fuite du gradient qui
peut se produire lorsque le softmax prend des valeurs extrêmes, en limitant la magnitude
du produit scalaire dont l’étendue des valeurs augmente naturellement avec la dimension.

Le **softmax**, opéré sur chaque ligne de la matrice, permet d’obtenir une pondération com-
posée de valeurs positives sommant à un, similaire à des **probabilités associées aux n mots**
d’entrée, pour chaque requête q

Les poids d’attention s’écrivent $ α = softmax( \frac{QK^T}{√ρ_w})$  et se somment à 1

La multiplication matricielle entre ces poids d’attention et $V$ consiste à réaliser des moyennes pondérées des v_j

 **Le produit scalaire entre requêtes et clefs génère**
**des poids d’attention qui permettent de pondérer les valeurs, de sorte à représenter ce mot**
**contextuellement aux autres mots de la phrase.**

L'intuition derrière cette formule est que si αij est proche de 1, le vecteur yi sera
fortement influencé par la valeur vj (et donc à une projection linéaire près par xj).

si αij est proche de 0, c’est que yi n’est pas lié à xj. 

les seuls paramètres
de ce mécanisme sont Wq,Wk et Wv de dimensions ρw ×ρw et leur rôle est de projeter
la séquence d’entrée dans trois espaces Q,K et V de sorte à capturer les dépendances
entre les mots. 

 Les concepts de « clef », « valeur » et « requête » proviennent de systèmes
de stockage des données

. Par exemple, lorsque l’on saisit une **requête** pour rechercher un
document dans une base de données, un moteur de recherche associe cette requête à un
ensemble de **clefs** enregistrés dans la base (titre du document, contenu, date, etc.), puis il
présente les meilleures correspondances de documents (**valeurs**)

Dans le Transformer, **le mécanisme d’attention est utilisé de deux façons différentes** :

1. dans des mécanismes d’**auto-attention** (self-attention
   * de sorte à **construire des repré-**
     **sentations contextuelles Xc des mots de la séquence d’entrée et Y c des mots de la séquence**
     **de sortie** 
2. dans le décodeur
   *  les **requêtes sont construites à partir des sorties**
     **contextuelles** Y c
   * **les clefs et valeurs sont construites à partir des entrées contextuelles**
     **Xc**
   * **Le décodeur fabrique ainsi des représentations de la séquence de sortie capturant ses**
     **dépendances avec la séquence d’entrée**, facilitant la prédiction du prochain mot

le
modèle dans son intégralité est en réalité composé de multiples mécanismes d’attention
opérés en parallèles, concaténés puis réutilisés sur de multiples couches. L’architecture
entière fait intervenir un grand nombre de perceptrons à plusieurs couches augmentant
significativement le nombres de paramètres et rendant non trivial l’entraînement du mo-
dèle

Le Transformer, plus précisément son **encodeur avec ses mécanismes d’auto-attention**,
constitue la brique de base de **BERT** [

L’idée principale est de pré-entraîner ce mécanisme d’attention sur
des tâches très générales ne requérant pas d’annotation particulière avant d’affiner les
paramètres du modèle sur des tâches spécifiques, telles que la détection d’entité nommée et
l’analyse de sentiment. 

Pour le **pré-entraînement**, **deux tâches** sont proposées : 

1. **la prédiction de mots masqués** 
   * on
     présente au modèle des phrases dont 15% des mots ont été remplacés par un vecteur
     spécial de masque et on optimise les paramètres du modèle de sorte à ce que les vecteurs
     de sortie associés aux masques soient les plus proches possible des vecteurs des mots
     cachés. 
2. la **prédiction de phrases successives**.
   * on tire aléatoirement 50% de paires de phrases qui se
     suivent dans un corpus et 50% de paires de phrases qui ne se suivent pas et on entraîne
     un classifieur à prédire si ces phrases se suivent à partir d’une représentation de sortie
     produite à partir d’un vecteur spécial de classification ajouté aux phrases d’entrée.



### [Comprendre le langage à l'aide de XLNet avec pré-formation autorégressive](https://ichi.pro/fr/comprendre-le-langage-a-l-aide-de-xlnet-avec-pre-formation-autoregressive-172103194027075)                    

**XLNet** surpasse le BERT sur 20 tâches de référence en PNL

XLNet exploite le meilleur de la modélisation de **langage autorégressif** (AR) et de **l'autoencodage** (AE), les deux objectifs de pré-formation les plus connus, tout en évitant leurs limites

Considéré comme l'un des développements les plus importants de 2019 en PNL, XLNet **combine le modèle de langage autorégressif**, Transformer-XL , **et la capacité bidirectionnelle** de BERT pour libérer la puissance de cet important outil de modélisation de langage

Pour la phase de pré-formation, les deux architectures les plus réussies sont la modélisation du langage autorégressif (AR) et l'autoencodage  (AE). 

1. ##### Modélisation du langage autorégressif (AR)

Dans les modèles AR conventionnels, le contexte unidirectionnel dans le sens avant ou arrière dans une séquence de texte est codé

 Il est utile pour les tâches NLP génératives qui génèrent un contexte dans le sens direct.

Cependant, AR échoue dans le cas où le contexte bidirectionnel doit être utilisé simultanément . Cela pourrait devenir problématique, en particulier avec la tâche de compréhension du langage en aval où des informations de contexte bidirectionnelles sont requises.

2. ##### Modèle de langage de codage automatique (AE)

Un modèle basé sur AE a la capacité de modéliser des contextes bidirectionnels en reconstruisant le texte original à partir d'une entrée corrompue ([MASK]). Le modèle AE est donc meilleur que le modèle AR lorsqu'il s'agit de mieux capturer le contexte bidirectionnel.

Un exemple notable d'AE est **[BERT](https://arxiv.org/pdf/1810.04805.pdf) qui est basé sur l'auto-encodage de débruitage**. 

il souffre d'un **écart de pré-entraînement-finetune** résultant de la dépendance entre les jetons masqués et ceux non masqués.

[MASK] utilisé dans l'étape de pré-formation est absent des données réelles utilisées aux tâches en aval, y compris l'étape de réglage fin. 

Pour les caractéristiques de dépendance d'ordre élevé et à longue portée en langage naturel, BERT simplifie à l'extrême le problème en supposant que les jetons prédits (masqués dans l'entrée) sont indépendants les uns des autres tant que les jetons non masqués sont donnés.

Alors que AR peut estimer la probabilité d'un produit avant ou arrière sous la forme d'une distribution de probabilité conditionnelle, **BERT ne peut pas modéliser la probabilité conjointe en utilisant la règle du produit en raison de son hypothèse d'indépendance pour les jetons masqués.**

3. ##### En quoi XLNet diffère-t-il des AR et AE conventionnels (BERT)?

Les auteurs de XLNet proposent de **conserver les avantages du modèle de langage AR tout en lui faisant apprendre du contexte bidirectionnel en tant que modèles AE (par exemple, BERT) pendant la phase de pré-formation**. 

**L'interdépendance entre les jetons sera préservée**, contrairement à BERT. 

Le nouvel objectif proposé est appelé "**Modélisation du langage de permutation**. "

Différent de BERT et d'autres transformateurs qui combinent l'incorporation de position et l'incorporation de contenu pour la prédiction, XLNet prédit la distribution du prochain jeton en prenant en compte la position cible z_t comme entrée. 

L'architecture **d'auto-attention à deux flux** est utilisée pour  résoudre les problèmes que pose le transformateur traditionnel; se compose de **deux types  d'attention personnelle**. 

1. la **représentation du flux de  contenu**
   * identique à l'auto-attention standard de Transformer qui prend  en compte à la fois le contenu (x_ {z_t}) et les informations de  position (z_t). 
2. la **représentation de requête**, 
   * **remplace  essentiellement le [MASK]** de BERT, appris par l'attention du flux de  requête pour prédire x_ {z_t} **uniquement avec des informations de  position mais pas son contenu**.
   * seules les informations de position du  jeton cible et les informations de contexte avant le jeton sont  disponibles.

Le résultat final de l'attention à deux flux est une **distribution de  prédiction sensible à la cible**. 

La principale différence entre XLNet et  BERT est que **XLNet n'est pas basé sur la corruption de données** comme le  fait BERT, il peut donc éviter les limitations de BERT résultant du  masquage

XLNet intègre un **schéma de codage relatif** et un **mécanisme de récurrence de segment** de [Transformer-XL](https://arxiv.org/pdf/1901.02860.pdf) pour capturer les dépendances qui sont plus éloignées que les RNN et Transformer. 

Le **codage  positionnel relatif** est appliqué en fonction de la séquence d'origine. 

Le **mécanisme de récurrence au niveau du segment** évite la fragmentation  de contexte représentée par le traitement de segment de longueur fixe.  Il permet de réutiliser des segments de phrase du passé avec le nouveau  segment. Le Transformer-XL réalise cela en incluant la récurrence au  niveau du segment dans les états masqués.

le Transformer standard contient des **informations de position dans les codages de position**, matrice U , avec intégration de position absolue. 

Transformateur-XL **code pour la distance relative dynamiquement dans le score de l' attention** en introduisant la matrice R . 

Dans le score d'attention de Transformer-XL, les 4 termes représentent respectivement l'adressage basé sur le contenu, le biais de position dépendant du contenu, le biais de contenu global et le biais de position global. 

Avec Transformer-XL, des articles de texte cohérents peuvent être générés et il y a également une accélération substantielle lors de l'évaluation par rapport aux RNN et au Transformer standard.

**Dans XLNet, Transformer-XL est inclus dans le cadre de pré-formation**. Le mécanisme de récurrence de Transformer-XL est ainsi incorporé dans le paramètre de permutation proposé dans XLNet pour réutiliser les états cachés des segments précédents.

 L'ordre de factorisation dans la permutation des segments précédents ne sera pas mis en cache et réutilisé à l'avenir. 

Seule la représentation du contenu du segment est conservée dans les états masqués.

**XLNet combine la capacité bidirectionnelle de BERT et la technologie autorégressive de Transformer-XL** pour réaliser une amélioration substantielle; il bat BERT dans plus d'une douzaine de tâches. 



### [Review: BioBERT paper](https://medium.com/@raghudeep/biobert-insights-b4c66fde8fa7)

The major contribution is a pre-trained bio-medical language representation model for various bio-medical text mining tasks. Tasks such as NER from Bio-medical data, relation extraction, question & answer in the  biomedical field.

BioBERT is a **contextualized language representation model, based on BERT**, a pre-trained model which is trained on different combinations of general & biomedical domain corpora.

Approach

* Initialize BioBERT with BERT pre-trained model trained on Wikipedia 2.5 billion words & Books Corpus 0.8 billion words. 
  * rather than taking a random initialization of weights, pre-trained weights from BERT model are taken. 
  * Transferring representations learned from previous corpora, it's similar to the transfer learning used for image data problems.
* pre-training on the domain data, here BioBERT is pre-trained on PubMed Abstracts 4.5 billion words & PMC Full-text articles 13.5 billion words.
* the pre-trained model is used to fine-tune on various biomedical text mining tasks like NER, question & answer, relation extraction.

The interesting part was that the **pre-training was not just on biomedical corpora but rather took different combinations of general & biomedical corpora** since their research objective was to figure out the performance of BERT with different combinations of corpora & the amount of data needed to pre-train the model.

Authors have used Word piece tokenizer (read sec 4.1 of the paper) as used by the BERT paper to mitigate OOV (out of the vocabulary) problem.

### [Approaches to Biomedical Text Mining with BERT](https://medium.com/geekculture/approaches-to-biomedical-text-mining-dca3408397b0)

BERT includes a pre-training step and a fine-tuning step.

##### (1) A pre-training step builds a generic language model.

BERT builds a language model by learning the context of words in a  left-to-right and right-to-left manner (**bidirectional**). 

during training, 15% of all words are masked (blanked out)  in sequences at random. 

The training process predicts the masked-out  word using a standard back propagation of errors method

BERT **also  learns the relationships between two sentences** by predicting whether the next sentence following the previous sentence is the actual *next* sentence or a random sentence. 

when choosing two  sentences A and B for each training sample, 50% of the time, B is the  actual next sentence that follows A, and 50% of the time it’s a random  sentence from the text. 

expensive labeling is not required in  either case

##### (2) A fine-tuning step with labeled data after pre-training with unlabeled data.

example of one of several fine-tuning, answering questions. 

**Fine-tuning** a supervised downstream task has the advantage of having to **learn only a few additional parameters with a relatively small, labeled dataset**.

BERT (fine-tuned on SQuAD) **learns two extra vectors that mark the beginning and the end of the answer span**. They are the start-token classifier and the end-token classifier that demarcate the answer (from the paragraph).

##### Applying BERT to the Biomedical Domain — BioBERT

The word distribution and context in the general text domain (WikiPedia and a large collection of books) is quite different from the biomedical domain (PubMed) and thus fine-tuning or adaptation is required (the architecture remains the same, however). Medical literature has a preponderance of medical terms; proper nouns (e.g., BRCA1, c.248T>C) and terms (e.g., transcriptional, antimicrobial), which are readily understood by biomedical researchers.

##### Pre-training BioBERT

Step 1, **initialize BioBERT with weights from BERT** (**transfer learning**).

Step 2, **BioBERT is pre-trained on biomedical domain text** (PubMed abstracts and PebMed Central full-text articles).



###  [ILLUSTRATION DE BERT](https://lbourdois.github.io/blog/nlp/BERT/) 

#####  1.1 Récapitulatif sur le word embeddings 

La pratique a fait émergée que c’était une excellente idée d’utiliser des **embeddings pré-entrainés sur de grandes quantités de données textuelles** au lieu de les former avec le modèle sur ce qui était souvent un petit jeu de données. Il est donc devenu possible de télécharger une liste de mots et leurs embeddings générées par le pré-entraînement avec Word2Vec ou GloVe. V

La structure encodeur-décodeur du Transformer le rend très efficace pour la traduction automatique. 

#####  **1.5 L’Open AI Transformer (Pré-entraînement d’un décodeur de Transformer pour la modélisation du langage)** 

nous n’avons pas besoin d’un Transformer complet pour adopter l’apprentissage par transfert dans le cadre de taches de NLP. Nous pouvons nous contenter du decodeur du Transformer. 

Le décodeur est un bon choix parce que c’est un choix naturel pour la modélisation du langage (prédire le mot suivant). En effet il est construit pour masquer les futurs tokens – une fonction précieuse lorsqu’il génère une traduction mot à mot.

Le modèle empile douze couches de décodeurs. Puisqu’il n’y a pas d’encodeur, les couches de décodeurs n’ont pas la sous-couche d’attention encodeur-décodeur comme dans le Transformer classique. Ils ont cependant toujours la couche d’auto-attention.

nous pouvons procéder à l’entraînement du modèle sur la même tâche de modélisation du langage : prédire le mot suivant en utilisant des ensembles de données massifs (sans label). 

L’entraînement est réalisé sur 7.000 livres car ils permettent au modèle d’apprendre à associer des informations connexes

 l’OpenAI Transformer est pré-entrainé et que ses couches ont été ajustées pour gérer raisonnablement le langage, nous pouvons commencer à l’utiliser pour des tâches plus spécialisées. 

GPT-2

2. ##### BERT : du décodeur à l’encodeur 

#####  **2.1 Architecture du modèle** 

L’article original présente deux tailles de modèles pour BERT :

- BERT BASE de taille comparable à celle de l’OpenAI Transformer afin de comparer les performances.
- BERT LARGE, un modèle beaucoup plus grand qui a atteint l’état de l’art des résultats rapportés dans l’article.

Les deux modèles BERT ont un grand nombre de **couches d’encodeurs** (appellées **Transformer Block** dans l’article d’origine) :

* 12 pour la version de base
* 24 pour la version large. 

Ils ont également **des réseaux feedforward** plus grands (768 et 1024 unités cachées respectivement) et plus de **têtes d’attention** (12 et 16 respectivement) que la configuration par défaut dans l’implémentation initial du Transformer 

#####  2.2 Entrées du modèle 

Le premier token d’entrée est un jeton spécial [CLS]

Tout comme l’encodeur du Transformer, BERT prend une séquence de mots en entrée qui remonte dans la pile**. Chaque couche applique l’auto-attention et transmet ses résultats à un réseau feed-forward, puis les transmet à l’encodeur suivant**

Trouver la bonne manière d’entraîner une pile d’encodeurs est un  obstacle complexe que BERT résout en adoptant un concept de « **modèle de  langage masqué** » (Masked LM en anglais) tiré de la littérature  antérieure (il s’agit d’une **Cloze task**).

* prendre aléatoirement 15% des tokens en entrée puis à masquer 80% d’entre eux, en remplacer 10% par un autre token complètement aléatoire (un autre mot) et de ne rien faire dans le cas des 10% restant
* objectif: que le modèle prédise correctement le token original modifié (via la perte d’entropie croisée)
* Le modèle est donc **obligé de conserver une représentation contextuelle** distributionnelle de chaque jeton d’entrée.

Afin d’améliorer BERT dans **la gestion des relations** existant **entre plusieurs phrases**, le processus de pré-entraînement comprend une tâche supplémentaire : étant donné deux phrases (A et B), B est-il susceptible d’être la phrase qui suit A, ou non ?

#####  2.3 Sorties du modèle 

nous nous concentrons uniquement sur la sortie de la première position (à laquelle nous avons passé le jeton spécial [CLS]).

Ce vecteur peut maintenant être utilisé comme entrée pour un classifieur de notre choix. L’article obtient d’excellents résultats en utilisant simplement **un réseau neuronal à une seule couche comme classifieur**.

en cas de plusieurs labels: modifier le réseau du classifieur pour avoir plus de neurones de sortie qui passent ensuite par la couche softmax.

#####  2.4 Modèles spécifiques à une tâche 

 les auteurs de BERT précisent les approches de fine-tuning appliquées pour quatre tâches de NLP différentes

1. classification
2. tests de logique
3. QA
4. NER

#####  2.5 BERT pour l’extraction de features 

L’approche fine-tuning n’est pas l’unique manière d’utiliser BERT. Tout comme ELMo, vous pouvez utiliser BERT pré-entrainé pour créer des word embeddings contextualisés. Vous pouvez ensuite intégrer ces embeddings à votre modèle existant.

##### remarques

BERT ne considère pas les mots comme des tokens. Il regarde plutôt les WordPieces (par exemple : playing donne play + ##ing).



### [ILLUSTRATION DU WORD EMBEDDING ET DU WORD2VEC](https://lbourdois.github.io/blog/nlp/word_embedding/)               



Au lieu de regarder seulement deux mots avant le mot cible, nous pouvons aussi regarder deux mots après lui. C’est ce qu’on appelle une architecture de **continuous bag of words**.

 Au lieu de deviner un mot en fonction de son contexte (les mots avant et après), cette autre architecture essaie de deviner les mots voisins en utilisant le mot courant. Cette méthode s’appelle l’architecture **skipgram**.

Pour générer des embeddings de haute qualité, nous pouvons passer d’un modèle de la prédiction d’un mot voisin à un modèle qui prend le mot d’entrée et le mot de sortie, et sort un score indiquant s’ils sont voisins ou non (0 pour « non voisin », 1 pour « voisin »).

Nous passons d’un réseau neuronal à un modèle de régression logistique qui est ainsi beaucoup plus simple et beaucoup plus rapide à calculer.

 Mais il y a une faille à combler. Si tous nos exemples sont positifs (cible : 1), nous nous ouvrons à la possibilité d’un modèle qui renvoie toujours 1 – atteignant 100% de précision, mais n’apprenant rien et générant des embeddings de déchets.

 nous devons introduire des échantillons négatifs dans notre ensemble de données, c’est à dire des échantillons de mots qui ne sont pas voisins. Notre modèle doit retourner 0 pour ces échantillons. (**negative sampling**) -> **Skipgram with Negative Sampling**

11. ##### Processus d’entraînement de Word2vec 

Au début de la phase d’entraînement, nous créons deux matrices – une **Embedding matrix** et une **Context matrix**. 

Ces deux matrices ont un embedding pour chaque mot de notre vocabulaire (*vocab_size* est donc une de leurs dimensions). 

La seconde dimension est la longueur que nous voulons que chaque vecteur d’embedding soit (une valeur généralement utilisée de *embedding_size* est 300, mais nous avons regardé un exemple de 50 plus tôt dans ce post).

au début de l’entraînement, nous initialisons ces matrices avec des valeurs aléatoires.

A chaque étape de l’entraînement, nous prenons un exemple positif et les exemples négatifs qui y sont associés

Nous procédons à la recherche de leurs embeddings. Pour le mot d’entrée, nous regardons dans l’Embedding matrix. Pour les mots de contexte, nous regardons dans la Context matrix.

 nous effectuons le produit scalaire de l’embeddings d’entrée avec chacun des embeddings de contexte.

transformer ces scores en quelque chose qui ressemble à des probabilités. Nous avons besoin qu’ils soient tous positifs et qu’ils aient des valeurs entre zéro et un. Pour cela, nous utilisons la fonction sigmoïde.

La taille de la fenêtre et le nombre d’échantillons négatifs sont deux hyperparamètres clés dans le processus d’entraînement de word2vec.

Une heuristique est que des fenêtres de petite taille (2-15) conduisent à des embeddings avec des scores de similarité élevés entre deux embeddings. Cela signifie que les mots sont interchangeables 

Des fenêtres de plus grande taille (15-50, ou même plus) mènent à des embeddings où la similarité donne une indication sur la parenté des mots

###  [ILLUSTRATION DU TRANSFORMER](https://lbourdois.github.io/blog/nlp/Transformer/) 

Transformer, un modèle qui utilise l’attention pour augmenter la vitesse à laquelle ces modèles peuvent être entraînés

 Le Transformer surpasse le modèle de traduction automatique de Google dans des tâches spécifiques

 **Le plus grand avantage, vient de la façon dont le Transformer se prête à la parallélisation.**

1. ##### apperçu haut niveau

un composant d’encodage, un composant de décodage et des connexions entre eux.

Le composant d’encodage est une pile **d’encodeurs** (l’article empile six encodeurs les uns sur les autres)

Le composant de décodage est une pile de **décodeurs** du même nombre.

**Les encodeurs**

* tous **identiques mais ne partagent pas leurs poids**
* chacun divisé en 2 sous-couches :
  1. Les entrées de l’encodeur passent d’abord par **une couche  d’auto-attention** 
     * **aide l’encodeur à regarder les autres  mots dans la phrase d’entrée** lorsqu’il code un mot spécifique
  2. Les sorties de la couche d’auto-attention sont transmises à un **réseau feed-forward**. 
     * Le même réseau feed-forward **appliqué indépendamment à chaque encodeur**.

**Le décodeur**

* possède ces 2 couches, mais entre elles se trouve une **couche d’attention qui aide le décodeur à se concentrer sur les parties pertinentes de la phrase d’entrée** **(encoder-decoder attention**; comme dans les modèles seq2seq).

#####  **2. Les tenseurs** 

 nous commençons par transformer chaque mot d’entrée en vecteur à l’aide d’un algorithme d’embedding.

**L’embedding n’a lieu que dans l’encoder inférieur.** Le point commun à **tous les encodeurs est qu’ils reçoivent une liste de vecteurs de la taille 512.** Dans l’encoder du bas cela serait le word embeddings, mais dans les autres encodeurs, ce serait la sortie de l’encodeur qui serait juste en dessous.

La **taille de la liste** est un hyperparamètre que nous pouvons définir. Il s’agirait essentiellement de **la longueur de la phrase la plus longue** dans notre ensemble de données d’entraînement.

Après avoir enchassé les mots dans notre séquence d’entrée, chacun d’entre eux traverse chacune des deux couches de l’encodeur.

 dans chacune des positions, le mot circule à travers son propre chemin dans l’encodeur

Il y a des **dépendances entre ces chemins dans la couche d’auto-attention.**

**La couche feed-forward n’a pas ces dépendances** et donc les différents chemins peuvent être exécutés en parallèle lors de cette couche.

#####  **3. L’encodage** 

un encodeur reçoit une liste de vecteurs en entrée. Il traite cette liste en passant ces vecteurs dans une couche d’auto-attention, puis dans un réseau feed-forward, et enfin envoie la sortie vers le haut au codeur suivant.

 Le mot à chaque position passe par un processus d’auto-attention. Ensuite, chacun d’eux passe par un réseau feed-forward (le même réseau feed-forward pour chaque vecteur mais chacun le traverse séparément). 

#####  **4. Introduction à l’auto-attention** 

Au fur et à mesure que le modèle traite chaque mot (chaque position dans la séquence d’entrée, **l’auto-attention lui permet d’examiner d’autres positions dans la séquence d’entrée à la recherche d’indices qui peuvent aider à un meilleur codage pour ce mot.**

L’auto-attention est la méthode que le Transformer utilise pour  **améliorer la compréhension du mot** qu’il est en train de traiter en  fonction des autres mots pertinents.

5. ##### L’auto-attention en détail 

*1ère étape*: **créer 3 vecteurs à partir de chacun des vecteurs d’entrée** xi de l’encodeur (dans ce cas, l’embedding de chaque mot).

Chaque vecteur d’entrée xi est utilisé de 3 manières différentes dans l’opération d’auto-attention :

1. Il est **comparé à tous les autres vecteurs** pour établir les **pondérations pour sa propre production** yi ->  forme le **vecteur de requête** (**Query**)
2. Il est **comparé à tous les autres vecteurs** pour établir les **pondérations pour la sortie** du j-ème vecteur yj -> forme le **vecteur de clé** (**Key**).
3. Il est **utilisé comme partie de la somme pondérée** pour **calculer chaque vecteur de sortie** une fois que les pondérations ont été établies -> forme le **vecteur de valeur** (**Value**).

Ces vecteurs sont créés en multipliant l’embedding par trois matrices que nous avons formées pendant le processus d’entraînement

ces nouveaux vecteurs sont de plus petite dimension que le vecteur d’embedding; Ils n’ont pas besoin d’être plus petits. C’est un choix d’architecture pour rendre la computation des têtes d’attentions constante.

*2ème étape*:  calculer  un score. Example: calcul de l’auto-attention pour le 1er mot; il faut noter chaque mot de la phrase d’entrée par rapport à ce mot.  **Le score détermine le degré de concentration à placer sur les autres  parties de la phrase d’entrée** au fur et à mesure que nous codons un mot à une certaine position.

score est calculé en prenant le **produit scalaire du vecteur de requête avec le vecteur clé** du mot que nous évaluons. Donc, si nous traitons l’auto-attention pour le mot en position #1, le premier score serait le produit scalaire de q1  et k1. Le deuxième score serait le produit scalaire de q1 et k2.

*3ème et 4ème étape*: diviser les scores  par la racine carrée de la dimension des vecteurs clés utilisés -> permet d’obtenir des gradients plus stables.

softmax peut être sensible à de très grandes valeurs d’entrée -> tue le gradient et ralentit l’apprentissage, ou l’arrête complètement. 

la valeur moyenne du produit scalaire augmente avec la dimension de l’embedding, il est utile de redimensionner un peu le produit scalaire pour empêcher les entrées de la fonction softmax de devenir trop grandes.

Il pourrait y avoir d’autres valeurs possibles que la racine carrée de la dimension, mais c’est la valeur par défaut.

**Softmax** permet de normaliser les scores pour qu’ils soient tous positifs et somment à 1.

Ce score softmax détermine **à quel point chaque mot sera exprimé à sa  position**. Il est donc logique que le mot à sa position aura le score  softmax le plus élevé, mais **le score des autres mots permet de  déterminer leur pertinence par rapport au mot traité**.

*5ème étape*. **multiplier chaque vecteur de valeur par le score softmax** (en vue de les additionner) -> garder intactes les valeurs du ou des mots sur lesquels nous voulons nous concentrer, et de **noyer les mots non pertinents** (en les multipliant par de petits nombres comme 0,001, par exemple).

*6ème étape*. **résumer les vecteurs de valeurs pondérées**. Ceci **produit la sortie de la couche d’auto-attention à cette position** 

Les vecteurs zi résultants peuvent être envoyés au réseau feed-forward. En pratique cependant, ce calcul est effectué sous forme de matrice pour un traitement plus rapide

6. ##### Les matrices de calcul de l’auto-attention 

*1ère étape*. **calculer les matrices Requête, Clé et Valeur**. concaténer les embeddings dans une matrice X et la multiplier par les matrices de poids que nous avons entraînés 

*étapes 2 à 6*. peuvent être concaténées en 1 formule pour calculer les sorties de la couche  d’auto-attention.

 7. ##### La bête à plusieurs têtes

**Au lieu d’exécuter une seule fonction d’attention** les auteurs de l’article ont trouvé avantageux de **projeter linéairement les requêtes, les clés et les valeurs h fois avec différentes projections linéaires** apprises sur les dimensions dk, dk et dv, respectivement.

Ce mécanisme est appelé « **attention multi-têtes** ». Cela **améliore les performances de la couche d’attention** de deux façons :

1. élargit la capacité du modèle à se concentrer sur différentes positions
   * « Marie a donné des roses à Susane »: « donné » a des relations différentes aux différentes parties de la phrase. « Marie » exprime qui fait le don, « roses » exprime ce qui est donné, et « Susane » exprime qui est le destinataire. 
   * **En une seule opération d’auto-attention, toutes ces informations ne font que s’additionner** -> Si c’était Suzanne qui avait donné les roses plutôt que Marie, le vecteur de sortie zdonné serait le même, même si le sens a changé.
2. donne à la couche d’attention de **multiples « sous-espaces de représentation »**. 
   * avec l’attention à plusieurs têtes, nous n’avons pas seulement un, mais **plusieurs ensembles de matrices de poids** Query/Key/Value (le Transformer utilise huit têtes d’attention, donc nous obtenons huit ensembles pour chaque encodeur/décodeur)
   * chacun de ces ensembles est **initialisé au hasard**. 
   * après l’entraînement, chaque ensemble est utilisé pour projeter les embedding d’entrée (ou les vecteurs des encodeurs/décodeurs inférieurs) dans un **sous-espace de représentation différent.**

Si nous faisons le même calcul d’auto-attention que nous avons décrit ci-dessus, huit fois avec des matrices de poids différentes, nous obtenons huit matrices Z différentes.

mais la couche de feed-forward attend une matrice unique (un vecteur pour chaque mot). 

pour condenser ces huit éléments en une seule matrice: **concaténer** les matrices puis les multiplier par **une matrice de poids supplémentaire** WO.

#####  **8. Le codage positionnel** 

 façon de **rendre compte de l’ordre des mots dans la séquence d’entrée.**

 le Transformer **ajoute un vecteur à chaque embedding** d’entrée. Ces vecteurs suivent un modèle spécifique que le modèle apprend ce qui l’aide à déterminer la position de chaque mot (ou la distance entre les différents mots dans la séquence). L’intuition ici est que l’ajout de ces valeurs à l’embedding **fournit des distances significatives entre les vecteurs d’embedding** une fois qu’ils sont projetés dans les vecteurs Q/K/V (puis pendant l’application du produit scalaire).

différentes méthodes possibles pour le codage positionnel; e.g. les valeurs de la moitié gauche sont générées par une fonction (qui utilise le sinus), et la moitié droite est générée par une autre fonction (qui utilise le cosinus). Ils sont ensuite concaténés pour former chacun des vecteurs d’encodage positionnel. 

#####  **9. Les connexions résiduelles** 

 **chaque sous-couche (auto-attention, feed-forward) dans chaque codeur a une *connexion résiduelle* autour de lui et est suivie d’une étape de *normalisation*.**

Cela vaut également pour les sous-couches du décodeur.

10. ##### Le decodeur 

L’encoder commence par traiter la séquence d’entrée. 

La **sortie de l’encoder supérieur** est ensuite transformée en un ensemble de **vecteurs d’attention K et V**. 

Ceux-ci doivent être **utilisés par chaque décodeur dans sa couche « attention encodeur-décodeur »** qui permet au decodeur de se concentrer sur les endroits appropriés dans la séquence d’entrée 

Chaque étape de la phase de décodage produit un élément de la séquence de sortie

Les étapes suivantes répètent le processus jusqu’à ce qu’un symbole spécial indique au décodeur que le Transformer a complété entièrement la sortie. 

**La sortie de chaque étape (mot ici) est envoyée au décodeur le plus bas pour le traitement du mot suivant**. 

 tout comme pour les entrées encodeur, nous « embeddons » et **ajoutons un codage positionnel à ces entrées décodeur** pour indiquer la position de chaque mot.

Les couches **d’auto-attention du décodeur** fonctionnent d’une manière légèrement différente de celle de l’encodeur.

* la couche d’**auto-attention** ne peut s’occuper **que des positions antérieures** dans la séquence de sortie. 
  * fait en masquant les positions futures (en les réglant sur -inf) avant l’étape softmax du calcul de l’auto-attention.

* la couche « **Attention encodeur-décodeur** » fonctionne comme une auto-attention à plusieurs têtes, sauf qu’elle crée sa **matrice de requêtes à partir de la couche inférieure**, et prend la **matrice des clés et des valeurs à la sortie de la pile encodeur**. 

 11. ##### Les couches finales : linéaire et sofmax

La pile de decodeurs délivre un vecteur de float. 

Comment le transformer en mots ? C’est le travail de la couche Linéaire qui est suivie d’une couche Softmax.

La **couche linéaire** est un simple réseau neuronal entièrement connecté qui **projette le vecteur produit par la pile de decodeurs dans un vecteur beaucoup (beaucoup) plus grand appelé vecteur logits**.

Supposons que notre modèle connaisse 10 000 mots anglais uniques (le « vocabulaire de sortie » de notre modèle) qu’il a appris de son ensemble de données d’entraînement. Cela rendrait le vecteur logit large de 10 000 cellules, **chaque cellule correspondant au score d’un mot unique**. C’est ainsi que nous interprétons la sortie du modèle suivie de la couche linéaire.

La **couche softmax** **transforme ensuite ces scores en probabilités** (tous positifs dont la somme vaut 1). La cellule ayant la probabilité la plus élevée est choisie et le mot qui lui est associé est produit comme sortie pour ce pas de temps.

#####  **12. L’entraînement** 

Pendant l’entraînement, un modèle non entraîné passerait exactement par le même processus. Mais puisque nous l’entraînons sur un ensemble de données d’entraînement labellisé, nous pouvons comparer sa sortie avec la sortie correcte réelle.

Une fois que nous avons défini notre vocabulaire de sortie, nous pouvons utiliser un vecteur de la même largeur pour indiquer chaque mot de notre vocabulaire. C’est ce qu’on appelle aussi le **one-hot encoding**

#####  **13. La fonction de perte** 

Comment comparer deux distributions de probabilités ? Nous  soustrayons simplement l’une à l’autre. Pour plus de détails, voir  l’entropie croisée et la divergence de Kullback-Leibler.

 Par exemple en entrée : « Je suis étudiant » et comme résultat attendu : « I am a student ». Ce que cela signifie vraiment, c’est que nous voulons que notre modèle produise successivement des distributions de probabilités où :

* Chaque distribution de probabilité est représentée par un vecteur de largeur vocab_size (6 dans notre exemple, mais de façon plus réaliste un nombre comme 3 000 ou 10 000)
* La première distribution de probabilités a la probabilité la plus élevée à la cellule associée au mot « I »
* La deuxième distribution de probabilité a la probabilité la plus élevée à la cellule associée au mot « am »
* Et ainsi de suite jusqu’à ce que la cinquième distribution de sortie indique ‘’, auquel est également associée une cellule du vocabulaire à 10 000 éléments



### [LE SEQ2SEQ ET LE PROCESSUS D’ATTENTION](https://lbourdois.github.io/blog/nlp/Seq2seq-et-attention/)               

Un sequence-to-sequence model est un modèle qui prend une séquence d’éléments (mots, lettres, caractéristiques d’une image…etc) et en sort une autre séquence. 

Sous le capot, le modèle est composé d’un encodeur et d’un décodeur.

L’ **encodeur** traite chaque élément de la séquence d’entrée. Il compile les informations qu’il capture dans un vecteur (appelé **context**). Après avoir traité toute la séquence d’entrée, l’encodeur envoie le context au **décodeur**, qui commence à produire la séquence de sortie item par item.

L’encodeur et le décodeur ont tendance à être tous deux des réseaux neuronaux récurrents.

Le **vecteur de contexte s’est avéré être un goulot d’étranglement** pour ces types de modèles. Il était donc difficile pour les modèles de composer avec de longues phrases. Une solution a été proposée dans Bahdanau et al., 2014 et Luong et al., 2015. Ces articles introduisirent et affinèrent une technique appelée « **Attention** », qui améliora considérablement la qualité des systèmes de traduction automatique. L’attention **permet au modèle de se concentrer sur les parties pertinentes de la séquence d’entrée si nécessaire.**

Cette **capacité d’amplifier le signal de la partie pertinente** de la séquence d’entrée permet aux modèles d’attention de produire de meilleurs résultats que les modèles sans attention.

Un modèle d’attention diffère d’un sequence-to-sequence model classique de deux façons principales :

1. **l’encodeur transmet beaucoup plus de données au decodeur**. 
   * Au lieu de passer le dernier état caché de l’étape d’encodage, **l’encodeur passe tous les états cachés au decodeur **
2.  un décodeur d’attention fait une étape supplémentaire avant de produire sa sortie. Pour se concentrer sur les parties de l’entrée qui sont pertinentes, le décodeur
   * **regarde l’ensemble des états cachés de l’encodeur** qu’il a reçu (chaque état caché de l’encoder est le plus souvent associé à un certain mot dans la phrase d’entrée).
   * donne un **score à chaque état caché** 
   * **multiplie chaque état caché par son score** attribué via softmax (amplifiant ainsi les états cachés avec des scores élevés, et noyant les états cachés avec des scores faibles)

 Le « scorage » se fait à chaque pas de temps (nouveau mot) du côté du décodeur.

 le modèle n’associe pas seulement le premier mot de la sortie avec le premier mot de l’entrée. En fait, il a appris pendant la phase d’entrainement la façon dont sont liés les mots dans cette paire de langues (le français et l’anglais dans notre exemple). 

###  [LES RNN, LES LSTM, LES GRU ET ELMO](https://lbourdois.github.io/blog/nlp/RNN-LSTM-GRU-ELMO/) 

Les RNN (recurrent neural network ou réseaux de neurones récurrents en français) sont des réseaux de neurones qui ont jusqu’à encore 2017/2018, été majoritairement utilisé dans le cadre de problème de NLP.

Cette architecture possède un problème. **Lorsque la séquence à traiter est trop longue, la rétropropagation du gradient de l’erreur peut soit devenir beaucoup trop grande et exploser, soit au contraire devenir beaucoup trop petite**. **Le réseau ne fait alors plus la différence entre une information qu’il doit prendre en compte ou non**. Il se trouve ainsi dans l’incapacité d’apprendre à long terme. 

solution proposée par Les LSTM: prendre en compte un **vecteur mémoire** **via un système de 3 portes (gates) et 2 états**

- **Forget** **gate** (capacité à oublier de l’information, quand celle-ci est inutile)
- **Input** **gate** (capacité à prendre en compte de nouvelles informations utiles)
- **Output** **gate** (quel est l’état de la cellule à l’instant t sachant la forget et la input gate)
- **Hidden** **state** (état caché)
- **Cell** **state** (état de la cellule)

**GRU** = une variante des LSTM; structure plus simple que les LSTM car moins de paramètres entrent en jeu. **2 portes et 1 état**:

* **Reset** **gate** (porte de reset)
* **Update** **gate** (porte de mise à jour)
* **Cell** **state** (état de la cellule)

En pratique, les GRU et les LSTM permettent d’obtenir des résultats comparables. **L’intérêt des GRU par rapport aux LSTM étant le temps d’exécution qui est plus rapide puisque moins de paramètres doivent être calculés.**



#####  ELMo (l’importance du contexte)

Embeddings from Language Models = basé sur un LSTM bidirectionnel

un mot peut avoir plusieurs sens selon la manière dont où il est utilisé; Pourquoi ne pas lui donner un embedding basé sur le contexte dans lequel il est utilisé ? A la fois pour capturer le sens du mot dans ce contexte ainsi que d’autres informations contextuelles ->**contextualized word-embeddings**.

Au lieu d’utiliser un embedding fixe pour chaque mot, **ELMo examine l’ensemble de la phrase avant d’assigner une embedding** à chaque mot qu’elle contient. Il utilise un **LSTM bidirectionnel formé sur une tâche spécifique pour pouvoir créer ces embedding**.

ELMo a constitué **un pas important vers le pré-entraînement** dans le contexte du NLP. En effet, nous pouvons l’entraîner sur un ensemble massif de données dans la langue de notre ensemble de données, et ensuite nous pouvons **l’utiliser comme un composant dans d’autres modèles** qui ont besoin de traiter le langage.

Plus précisément, **ELMo est entraîné à prédire le mot suivant dans une séquence de mots** – une tâche appelée modélisation du langage (Language Modeling). C’est pratique car nous disposons d’une grande quantité de données textuelles dont un tel modèle peut s’inspirer sans avoir besoin de labellisation.

​                                                                                                                              

ELMo va même plus loin et forme un **LSTM bidirectionnel**

ELMo propose **l’embedding contextualisé en regroupant les états cachés (et l’embedding initial)** d’une certaine manière (concaténation suivie d’une sommation pondérée).



### [ILLUSTRATION DU GPT2](https://lbourdois.github.io/blog/nlp/GPT2/)               

basé sur un Transformer entraîné sur un ensemble de données massif. 

le modèle de Transformer original est composé d’un encodeur et d’un  décodeur (chacun est une pile de ce que nous pouvons appeler des  transformer blocks). Cette architecture est appropriée parce que le  modèle s’attaque à la traduction automatique

##### **Une différence par rapport à BERT** 

**Le GPT-2 est construit à l’aide de blocs décodeurs**. 

**BERT, pour sa  part, utilise des blocs d’encodeurs**. 

l’une des principales différences entre les  deux est que **le GPT-2, comme les modèles de langage traditionnels,  produit un seul token à la fois**.

 Invitons par exemple un GPT-2 bien  entraîné à réciter la première loi de la robotique :  « A robot may not  injure a human being or, through inaction, allow a human being to come  to harm ».

La façon dont fonctionnent réellement ces modèles est qu’**après chaque token produit, le token est ajouté à la séquence des entrée**s. Cette nouvelle séquence devient l’entrée du modèle pour la prochaine étape. Cette idée est appelée « **autorégression** » et a permis aux RNN d’être efficaces.

**Le GPT2 et certains modèles plus récents comme TransformerXL et XLNet sont de nature autorégressive**; **BERT ne l’est pas.**

C’est un compromis. **En perdant l’autorégression, BERT a acquis la  capacité d’incorporer le contexte des deux côtés d’un mot** pour obtenir  de meilleurs résultats. 

**XLNet ramène l’autorégression tout en trouvant  une autre façon d’intégrer le contexte des deux côtés.**

 Un bloc encodeur de l’article d’origine peut recevoir des entrées jusqu’à une certaine longueur maximale (512 tokens). Si une séquence d’entrée est plus courte que cette limite, nous avons simplement à rembourrer le reste de la séquence. (ajouter <pad>)

**Une différence clé dans la couche d’auto-attention est qu’elle masque les futurs tokens – non pas en changeant le mot en [mask] comme BERT,  mais en interférant dans le calcul de l’auto-attention en bloquant les  informations des tokens qui sont à la droite de la position à calculer.**

Un bloc d’auto-attention normal permet à une position d’atteindre le  sommet des tokens à sa droite. L’auto-attention masquée empêche que cela se produise 

Le modèle OpenAI GPT-2 utilise uniquement ces blocs décodeurs.  très similaires aux blocs décodeurs d’origine, sauf qu’ils suppriment la deuxième couche d’auto-attention.

 Le GPT-2 peut traiter 1024 jetons. Chaque jeton parcourt tous les blocs décodeurs le long de son propre chemin. 

La façon la plus simple d’exécuter un GPT-2 entraîné est de lui  permettre de se promener de lui-même (ce qui est techniquement appelé  **generating unconditional samples**).   

Nous pouvons aussi le pousser à ce  qu’il parle d’un certain sujet (**generating interactive conditional  samples**). Dans le premier cas, nous pouvons simplement lui donner le  token de démarrage et lui faire commencer à générer des mots (le modèle  utilise *<|endoftext|>* comme token de démarrage. Appelons-le < s > à la place pour simplifier les graphiques).

Le token est traité successivement à travers toutes les couches, puis un vecteur est produit le long de ce chemin. 

GPT-2 a un paramètre appelé top-k que nous pouvons utiliser pour que le modèle considère des mots d’échantillonnage autres que le top mot (ce qui est le cas lorsque top-k = 1).



##### encodage de l'entrée

Comme dans d’autres modèles de NLP, le GPT-2 recherche l’**embedding** du mot d’entrée dans son embedding matrix (obtenue après entraînement).

Ainsi au début nous recherchons l’embedding du token de départ < s > dans la matrice. Avant de transmettre cela au premier bloc du  modèle, nous devons incorporer le **codage positionnel** (un signal qui  indique aux blocs l’ordre des mots dans la séquence). Une partie du  modèle entraîné contient une matrice ayant un vecteur de codage  positionnel pour chacune des 1024 positions de l’entrée.

 Envoyer un mot au premier bloc du Transformer, c’est rechercher son embedding et additionner le vecteur de codage positionnel pour la position #1.



##### voyage dans le bloc

Le premier bloc peut maintenant traiter le premier token en le  faisant passer d’abord par le processus d’auto-attention, puis par sa  couche feed forward. 

Une fois le traitement effectué, le bloc envoie le  vecteur résultant pour qu’il soit traité par le bloc suivant. 

Le  processus est identique dans chaque bloc mais **chaque bloc a des poids  qui lui sont propres dans l’auto-attention et dans les sous-couches du  réseau neuronal**.



##### l'auto-attention

 on attribue des notes à la pertinence de chaque mot du segment et additionne leur représentation vectorielle.

##### Le processus de l’auto-attention 

L’auto-attention est traitée le long du parcours de chaque token. Les composantes significatives sont **trois vecteurs** :

- **Query** : la requête est une **représentation du mot  courant**. Elle est **utilisée pour scorer le mot vis-à-vis des autres mots**  (en utilisant leurs clés).
- **Key** : les vecteurs clés sont comme des **labels**  pour tous les mots de la séquence. C’est **contre eux que nous nous  mesurons dans notre recherche de mots pertinents**.
- **Value** : les vecteurs de valeurs sont des  **représentations de mots réels**. Une fois que nous avons évalué la  pertinence de chaque mot, ce sont les valeurs que nous **additionnons pour représenter le mot courant**.

Une analogie grossière est de penser à la recherche dans un classeur. 

* La **requête (query) est le sujet que vous recherchez.** 
* Les **clés (key)  sont comme les étiquettes** des chemises à l’intérieur de l’armoire. 
* Lorsque vous faites correspondre la requête et la clé, nous enlevons le  contenu du dossier. **Le contenu correspond au vecteur de valeur (value)**.  

Sauf que vous ne recherchez pas seulement une valeur, mais un mélange de valeurs à partir d’un mélange de dossiers.

**Multiplier le vecteur de requête par chaque vecteur clé produit un score** pour chaque dossier (techniquement : le produit scalaire suivi de softmax).

**Nous multiplions chaque valeur par son score et sommons. Cela donne le résultat de notre auto-attention.**

Cette opération permet d’obtenir un vecteur pondéré



##### **Sortie du modèle**

Lorsque le bloc le plus haut du modèle produit son vecteur de sortie (le résultat de sa propre auto-attention suivie de son propre réseau feed-forward), le modèle **multiplie ce vecteur par la matrice d’embedding**.

chaque ligne de la matrice d’embedding correspond à un word embedding dans le vocabulaire du modèle. Le résultat de cette multiplication est interprété comme **un score pour chaque mot du vocabulaire** du modèle.

Nous pouvons simplement sélectionner le token avec le score le plus élevé (top_k = 1). Mais de meilleurs résultats sont obtenus si le modèle tient également compte d’autres termes. Ainsi, une bonne stratégie consiste à tirer au hasard un mot provenant du vocabulaire. **Chaque mot ayant comme probabilité d’être sélectionner, le score qui lui a été attribué** (de sorte que les mots avec un score plus élevé ont une plus grande chance d’être sélectionnés). Un terrain d’entente consiste à **fixer top_k à 40,** et à demander au modèle de prendre en compte les 40 mots ayant obtenu les scores les plus élevés.

Le modèle a alors terminé une itération aboutissant à l’édition d’un  seul mot. **Le modèle itère alors jusqu’à ce que le contexte entier soit  généré (1024 tokens) ou qu’un token de fin de séquence soit produit.**

en réalité, le GPT2 utilise le **Byte Pair Encoding** pour créer les tokens dans son vocabulaire = les tokens sont généralement des parties des mots



##### **Auto-attention (sans masking)**

 dans un bloc encoder. L’auto-attention s’applique en **trois** **étapes** principales :

1. **Création des vecteurs Query, Key et Value** pour chaque chemin.
2. Pour chaque token d’entrée, on utilise son **vecteur de requête pour lui attribuer un score par rapport à tous les autres vecteurs clés.**
3. **Sommation des vecteurs de valeurs après les avoir multipliés par leurs scores associés**. 



##### **Création des vecteurs Query, Key et Value** 

1. Pour le premier token, nous prenons sa requête et la comparons à  toutes les clés. Cela produit un score pour chaque clé. La **première  étape de l’auto-attention consiste à calculer les trois vecteurs pour  chaque token**  (en multipliant par la matrice de poids) (ignorons les têtes d’attention pour le moment) 
2. **multiplions sa requête par tous les autres vecteurs clés** pour obtenir un score pour chacun des quatre tokens.
3. **multiplier les scores par les vecteurs de valeurs**. Une valeur avec un score élevé constituera une grande partie du vecteur résultant une fois que nous les aurons additionnés.

Si nous faisons **la même opération pour chaque token, nous obtenons** **un vecteur représentant et tenant compte du contexte pour chacun d’eux**. Ces vecteurs sont ensuite présentés à la sous-couche suivante du bloc (le réseau de neurones feed-forward).



##### Auto-attention (avec masking)

identique à l’auto-attention, sauf à l’étape 2.

Supposons que le modèle n’a que deux tokens en entrée et que nous observons le deuxième token. Dans ce cas, les deux derniers tokens sont masqués. Le modèle attribue alors toujours aux futurs tokens un score de 0.

Ce « masquage » est souvent mis en œuvre sous la forme d’une matrice appelée **masque d’attention**.

A titre d’exemple, supposons avoir une séquence de quatre mots : « robot must obey orders ».

Sous forme matricielle, nous calculons les scores en multipliant une matrice de requêtes par une matrice clés 

le masque d’attention **règle les cellules que l’on veut **masquer sur **-inf ou un nombre négatif très important** (par exemple -1 milliard pour le GPT2) 

l’application de softmax produit les scores réels que nous utilisons pour l’auto-attention

#####  L’auto-attention masquée du GPT-2

**Le GPT-2 conserve les vecteurs clé et valeur des tokens qu’il a déjà traité afin de ne peut à avoir à les recalculer à chaque fois** qu’un nouveau token est traité. 

*Etape 1 : Création des vecteurs Query, Key et Value*

Chaque bloc d’un Transformer a ses propres poids. Nous nous servons de **la matrice de poids pour créer les vecteurs des requêtes, des clés et des valeurs**. Cela consiste en pratique à une **simple multiplication**.

 En multipliant le vecteur d’entrée par le vecteur de poids d’attention (et en ajoutant un vecteur de biais non représenté ici), on obtient les vecteurs clé, valeur et requête pour ce token. 

*Les têtes d'attention*

**L’auto-attention est menée plusieurs fois sur différentes parties des vecteurs Q,K,V.**

Séparer les têtes d’attention, c’est simplement **reconstruire le vecteur long sous forme de matrice.**

Le plus petit GPT2 possède 12 têtes d’attention. Il s’agit donc de la première dimension de la matrice remodelée 

 *Etape 2 : Scoring*

 *Etape 3 : Somme*

nous multiplions maintenant chaque valeur par son score, puis nous les additionnons pour obtenir le résultat de l’attention portée à la tête d’attention n°1 

 *Fusion des têtes d’attention*

Nous concaténons les têtes d’attention.

 *Etape 4 : Projection*

la 2ème grande matrice de poids qui projette les résultats des têtes d’attention dans le vecteur de sortie de la sous-couche d’auto-attention 

 *Etape 5 : Fully Connected Neural Network*

Le réseau neuronal entièrement connecté est l’endroit où le bloc traite son token d’entrée après que l’auto-attention a inclus le contexte approprié dans sa représentation. Il est composé de **deux couches**.

La première couche est quatre fois plus grande que le modèle (768x4 =  3072). Cela semble donner aux modèles  de Transformer une capacité de représentation suffisante pour faire face aux tâches qui leur ont été confiées jusqu’à présent.

La deuxième couche projette le résultat de la première couche dans la dimension du modèle (768 pour le petit GPT2). **Le résultat de cette multiplication est le résultat du bloc Transformer pour ce token.**

##### résumé

Chaque bloc a **son propre jeu de ces poids**. D’autre part, le modèle n’a qu’**une seule matrice d’embedding** de token et **une seule matrice de codage positionnel** 



### [bert-base-uncased](https://huggingface.co/bert-base-uncased?text=Paris+is+the+%5BMASK%5D+of+France)



BERT is a transformers model pretrained on a large corpus of English data in a self-supervised fashion.

it was pretrained with two objectives:

- **Masked language modeling** (MLM)
- **Next sentence prediction** (NSP)

This way, the model learns an inner representation of the English language that can then be used to extract features useful for downstream tasks: if you have a dataset of labeled sentences for instance, you can train a standard classifier using the features produced by the BERT model as inputs.

 **this model is primarily aimed at being fine-tuned on tasks that use the whole sentence (potentially masked) to make decisions, such as sequence classification, token classification or question answering. For tasks such as text generation you should look at model like GPT2.**

The inputs of the model are then of the form:

[CLS] Sentence A [SEP] Sentence B [SEP]

With probability 0.5, sentence A and sentence B correspond to two consecutive sentences in the original corpus and in the other cases, it's another random sentence in the corpus. Note that **what is considered a sentence here is a consecutive span of text usually longer than a single sentence. The only constrain is that the result with the two "sentences" has a combined length of less than 512 tokens.**

The details of the masking procedure for each sentence are the following:

15% of the tokens are masked.
In 80% of the cases, the masked tokens are replaced by [MASK].
In 10% of the cases, the masked tokens are replaced by a random token (different) from the one they replace.
In the 10% remaining cases, the masked tokens are left as is.
