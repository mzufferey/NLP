**Explanation regeneration** is the task of retrieving and combining two or more facts from an external knowledge source to reconstruct the explanation supporting a certain natural language hypothesis

Given a scientific hypothesis h (e.g., “Two sticks getting warm when rubbed together is an example ofa force producing heat”), the task of **explanation regeneration** consists in reconstructing the explana- tion supporting h, composing a sequence of atomic facts Eseq = f1, . . . , fn retrieved from external knowledge sources.

Explanation regeneration can be framed as a **multi-hop abductive inference problem**, where the goal is to construct the best explanation for a given natural language statement adopting multiple in-ference steps.

regeneration of natural language explanations is particularly chal- lenging for multi-hop inference models as it can lead to a phenomenon known as **semantic drift** – i.e., the composition of spurious inference chains caused by the tendency of drifting away from the original context in the hypothesis

 **multi-hop reasoning** – i.e. the integra- tion of supporting facts from different sources,

**Question Answering** (QA) is the task of inferring the answer for a natural language question in a given knowledge source

 **multi-hop reasoning**, i.e. the abil- ity of combining multiple information fragments from different sources.

Given a science question, **explanation reconstruction** consists in regenerating the gold ex- planation that supports the correct answer through the combination of a series of atomic facts.

constructing
long explanations is challenging due to **seman- tic drift** – i.e. the tendency of composing out-of- context inference chains as the number of hops increases 

 a **multi-hop inference** problem, where multiple pieces of evidence have to be aggregated to arrive at the final answer

 **semantic drift** (Khashabi et al., 2019; Fried et al., 2015) – i.e. the tendency of composing spurious inference chains leading to wrong conclusions

In general, an **explanation** can be seen as an answer to a how question formulated as follows: “How did the model arrive at the conclusion c starting from the problem formulation p?”

**Abductive reasoning** is inference to the most plausible explanation for incomplete observations

**abduction** is “the only logical operation which introduces any new ideas”, which contrasts with other types of inference such as entailment, that focuses on inferring only such information that is already provided in the premise

A **reasoning chain** is a sequence of sentences that logically connect the question to a fact relevant (or partially relevant) to giving a rea- sonably supported answer.

A **reasoning chain** is a sequence of sen- tences that logically connect the question to a fact relevant to determining the answer. Two adja- cent sentences in a reasoning chain should be intu- itively related: they should exhibit a shared entity or event, temporal structure, or some other kind of textual relation that would allow a human reader to connect the information they contain

Similar to multi-hop reasoning, **rule-based reason- ing** can also perform interpretable triple comple- tion, except that they give the corresponding rules instead of specific paths. Rule-based reasoning can be divided into two categories, namely, **neural- based models** and **rule mining models**. 

**combinatorial generalization**, that is, constructing new inferences, predictions, and behaviors from known building blocks

**structure** as the product of composing a set of known building blocks

“**Structured representations**” capture this composition (i.e., the arrangement of the elements) and

“**structured computations**” operate over the elements and their composition as a whole

**Relational reasoning**, then, involves manipulating structured representations of entities and relations, using rules for how they can be composed.

**relational inductive bias**. While not a precise, formal definition, we use this term to refer generally to **inductive biases** (Box 2) which impose constraints on relationships and interactions among entities in a learning process

**Locality** reflects that the arguments to the relational rule are those entities in close proximity with one another in the input signal’s coordinate space, isolated from distal entities.

**Translation invariance** reflects reuse of the same rule across localities in the input.

The problem of **Multi-Hop Natural Language inference** can be stated as follows: Given a hypothesis h (each natural language sentences), we say that **we may infer h if there exists a subset of supporting facts in a knowledge base** {f1, f2, . . .} ⊆ F of true statements **which would allow a human being to deduce h from {f1, f2, . . .}**. We call this set of facts an **explanation** for h

**Entailment** is a concept that refers to a specific kind of  relationship between two sentences. More specifically, entailment means  that if one sentence is true, then another sentence would also have to  be true: the second sentence would be entailed by the first sentence. Another way to prove entailment between two sentences is to  demonstrate that if the one sentence is false, then the other sentence  must also be false. Entailment is closely related to the concept of  logical consequence. Within logic, the idea that if A is true, then B  must be true too is nothing other than a form of entailment. 

**Explainable question answering** is the task of providing both answers to natural language questions, as well as detailed human-readable explanations justifying why those answers are correct

*  **retrieval methods** search for a single contiguous passage of text from a corpus or single fact in a knowledge base that provides an answer to a question. 
* For complex questions, a single passage often provides only part of the knowledge required to arrive at a correct answer, and an **inference model** must combine multiple facts from a corpus or knowledge base to infer the correct answer

, a frequent design goal of **multi-hop inference** algorithms is to use **the set of combined facts** as a human-readable explanation for why the model’s reasoning is correct.

**Self-attention**, sometimes called **intra-attention** is an attention mechanism **relating different positions of a single sequence in order to compute a representation of the sequence**

**End-to-end memory networks** are **based on a recurrent attention mechanism** instead of sequence- aligned recurrence and

At each step the model is **auto-regressive** [10], consuming the previously generated symbols as additional input when generating the next

An **attention function** can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key

We call our particular attention "**Scaled Dot-Product Attention"** 

* The input consists of queries and keys of dimension dk, and values of dimension dv
* compute the **dot products of the query with all keys, divide each by√ dk, and apply a softmax function to obtain the weights on the values**.
* n practice, we compute the **attention function on a set of queries simultaneously**, packed together into a matrix Q. The keys and values are also packed together into matrices K and V.
* The two most commonly used attention functions are additive attention [2], and dot-product (multi- plicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor (√ dk,)
  * dot-product attention is much faster and more space-efficient in practice, (optimized matrix multiplication code)
  * additive attention outperforms dot product attention without scaling for larger values of dk [3]. We suspect that for large values of dk, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients 4. To counteract this effect, we scale the dot products by

*Multi-Head Attention*

beneficial to linearly project the queries, keys and values h times with **different, learned linear projections** to dk, dk and dv dimensions, respectively

* On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding dv-dimensional output values. 
* These are concatenated and once again projected, resulting in the final values

* Multi-head attention **allows the model to jointly attend to information from different representation subspaces at different positions.**
  * single attention head, averaging inhibits this

*Applications of Attention in our Model*

Transformer uses multi-head attention in three different ways:

1. In "**encoder-decoder attention**" layers, 

   * the queries come from the previous decoder layer

   * memory keys and values come  from the output of the encoder

     -> allows every position in the decoder to attend over all positions in the input sequence

     -> mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models 

2. The **encoder contains self-attention** layers

   * all of the keys, values and queries come from the same place (the output of the previous layer in the encoder)
   * Each position in the encoder can attend to all positions in the previous layer of the encoder.

3. **self-attention layers in the decoder** 

   * allow each position in the decoder to attend to all positions in the decoder up to and including that position
   * prevent leftward information flow to preserve the auto-regressive property
   * implement this inside of scaled dot-product attention by masking out (setting to −∞) all values in the input of the softmax which correspond to illegal connections

*Position-wise Feed-Forward Networks*

, each of the layers in our encoder and decoder contains **a fully connected feed-forward network, which is applied to each position separately and identically**

* 2 linear transformations with a ReLU activation in between.

* linear transformations are the same across different positions, they use different parameters from layer to layer.
* = 2 convolutions with kernel size 1 
* dimensionality of input and output is dmodel = 512
*  inner-layer has dimensionality dff = 2048.

*Embeddings and Softmax*

* use **learned embeddings to convert the input tokens and output tokens to vectors of dimension dmodel**. 
* use the usual learned **linear transfor- mation and softmax function to convert the decoder output to predicted next-token probabilities**. In 
*  **same weight matrix between the two embedding layers and the pre-softmax linear transformation**, 
*  In the embedding layers, we multiply those weights by √ dmodel.

Positional Encoding

* no recurrence and no convolution: to make use of the order of the sequence,  must inject some information about the relative or absolute position of the tokens in the sequence
* add "**positional encodings**" **to the input embeddings at the bottoms of the encoder and decoder stacks**. 
* same dimension dmodel as the embeddings, so that the two can be summed
* many  possible choices of positional encodings, learned and fixed
* use sine and cosine functions of different frequencies
* experimented with using learned positional embeddings 
  * nearly identical results
  * chose the sinusoidal version: may allow the model to extrapolate to sequence lengths longer than the ones encountered during training

*self-attention*

**a self-attention layer connects all positions with a constant number of sequentially executed operations**, whereas a recurrent layer requires O(n) sequential operations

self-attention layers are faster than recurrent layers when the sequence length n is smaller than the representation dimensionality d, which is most often the case with sentence representations used by state-of-the-art models in machine translations

self-attention could yield **more interpretable** models. We inspect attention distributions

exhibit behavior related to the syntactic and semantic structure of the sentences.
5

A **cone program** is an optimization problem in which the objective is to minimize a linear function over the intersection of a subspace and a convex cone. every convex optimization problem can be expressed as a cone program

