#### Hybrid Autoregressive Solver for Scalable Abductive Natural Language Inference - Valentino et al. 2021

s intrin- sically not scalable, the cross-encoder archi- tectural paradigm is not suitable for efficient multi-hop inference on massive facts banks

maximise both accuracy and inference time, we propose a hybrid abductive solver that au- toregressively combines a dense bi-encoder with a sparse model of explanatory power, computed leveraging explicit patterns in the explanations. Our

**Explanation regeneration** is the task of retrieving and combining two or more facts from an external knowledge source to reconstruct the explanation supporting a certain natural language hypothesis

explanation regeneration on science ques-tions has been identified as a suitable benchmark for complex multi-hop and abductive inference ca- pabilities

Scientific explanations, in fact, require the articulation and integration of commonsense and scientific knowledge for the construction of long explanatory reasoning chains

since the structure of scientific explanations cannot be derived from the decomposition of the questions, the task requires the encoding of com- plex abstractive mechanisms for the identification of relevant explanatory knowledge

most of the exist-
ing approaches for explanation regeneration lever- age the power of the **cross-attention mechanism** in Transformers, training sequence classification mod- els on the task of composing relevant explanatory chains supervised via human-annotated explana- tions (Cartuyvels

While Transformers achieve state-of-the-art performance, the adoption of **cross-encoder architectures** makes abductive in- ference intrinsically inefficient and not scalable to massive corpora

state-of-the-art Transformer (Cartuyvels et al., 2020) requiring ≈ 9 seconds per question on

developing
new mechanisms to enable abductive inference at scale, maximising both accuracy and inference time in explanation regeneration

the construction of **abductive solvers** through **scalable bi-encoder architectures** (Reimers and Gurevych, 2019) to perform efficient **multi- hop inference** via Maximum Inner Product Search (MIPS) (

Given the complex- ity of abductive inference in the scientific domain, however, the adoption of bi-encoders alone is expected to lead to a drastic drop in performance since the cross-attention mechanism in Transformers can- not be leveraged to learn meaningful compositions of explanatory chains.

To tackle the **lack of cross- attention**, we hypothesise that the orchestration of latent and explicit patterns emerging in natural lan- guage explanations can enable the design of the **abstractive mechanisms** necessary for accurate re- generation while preserving the scalability intrinsic in bi-encoders	

we present SCAR
(for Scalable Autoregressive Inference), a **hybrid abductive solver** that autoregressively combines a **Transformer-based bi-encoder** with a **sparse model** that **leverages explicit patterns** in corpora of scien- tific explanations. Specifically,

Specifically, SCAR **integrates sparse and dense encoders to define a joint model of relevance and explanatory power and perform ab- ductive inference in an iterative fashion,** condition- ing the probability of retrieving a fact at time-step t on the partial explanation constructed at time-step t − 1

Given a scientific hypothesis h (e.g., “Two sticks getting warm when rubbed together is an example ofa force producing heat”), the task of **explanation regeneration** consists in reconstructing the explana- tion supporting h, composing a sequence of atomic facts Eseq = f1, . . . , fn retrieved from external knowledge sources.

Explanation regeneration can be framed as a **multi-hop abductive inference problem**, where the goal is to construct the best explanation for a given natural language statement adopting multiple in-ference steps.

To model the multi-hop nature of scientific explana- tions, we propose a hybrid abductive solver that per- forms inference autoregressively

Since similar scientific hypotheses require similar explanations (Valentino et al., 2021), we define ex- planatory power as a measure to capture explicit patterns in the explanations, quantifying the extent to which a given fact explains similar hypotheses in the corpus (additional

The probability of selecting a certain fact ft, therefore, depends jointly on its explanatory power and relevance with respect to the partially constructed explanation.

**explicit** explanatory patterns emerge in corpora of natural language explanations (Valentino et al., 2021) – i.e., facts describing scientific laws with **high explanatory power** (i.e., laws that explain a large set of phe- nomena such as gravity, or friction) are frequently **reused to explain similar** hypotheses. Therefore,

the explanatory power of a generic fact fi can be estimated by analysing explanations for similar hy- potheses in the corpus:

we hypothesise that this model
can be integrated within a hybrid abductive solver based on dense and sparse encoders (Sec.

we fine-tune a Sentence-BERT model using a bi-encoder architecture (

The bi-encoder adopts a siamese network to learn a joint embedding space for hypotheses and explanatory facts in the corpus

Following Sentence-BERT, we obtain fixed sized sentence embeddings by adding a mean-pooling operation to the output vectors of BERT

We employ a unique BERT model with shared parameters to learn a sentence encoder d(·) for both facts and hypotheses

At the cost of sacrificing the perfor- mance gain resulting from the cross-attention, the bi-encoder allows for efficient multi-hop inference through Maximum Inner Product Search (MIPS).

regeneration of natural language explanations is particularly chal- lenging for multi-hop inference models as it can lead to a phenomenon known as **semantic drift** – i.e., the composition of spurious inference chains caused by the tendency of drifting away from the original context in the hypothesis





#### Explainable Inference Over Grounding-Abstract Chains for Science Questions - Thayaparan et al. 2021

We propose an explainable inference approach for science questions by reasoning on ground- ing and abstract inference chains. This

question answering as a **natural lan- guage abductive reasoning problem**, construct- ing **plausible** explanations for each candidate answer and then selecting the candidate with the **best** explanation as the final answer.

ExplanationLP, **elicits explanations by constructing a weighted graph of relevant facts** for each candidate answer and employs a linear programming formalism designed to select the optimal subgraph of explanatory facts.

Current state-of-the- art (SOTA) approaches for answering questions in the science domain are dominated by transformer- based models

these approaches are black-box by nature lacking the capability of providing explanations for their predictions

**Explainable Science Question Answering** (XSQA) is often framed as a natural language abductive reasoning problem

**Abductive reasoning** represents a distinct inference process, known as inference to the best explanation, which starts from **a set of complete or incomplete observation**s to find the hypothesis, from **a set of plausible alternatives**, that **best** explains the observations. Several

XSQA solvers typically treat explanation gener-
ation as a **multi-hop graph traversal problem**: the solver attempts to compose multiple facts that connect the question to a candidate answer. These

multi-hop approaches have shown diminishing re- turns with an increasing number of hops; this phenomenon is due to semantic drift – i.e., as the number of aggregated facts increases, so does the probability of drifting out of context.

very long multi-hop reasoning chains are un- likely to succeed, emphasising the need for a richer representation with fewer hops and higher impor- tance to abstraction and grounding mechanisms

**Grounding Facts** that link generic or abstract con- cepts in a core scientific statement to specific terms occurring in question and candidate answer

The grounding process is followed by the identification of the **abstract facts**

A complete explanation for this question would require the composition of five facts to derive the correct answer successfully. However, it is pos- sible to reduce the global reasoning in two hops, modelling it with grounding and abstract facts. 

this work presents a novel approach that explicitly models abstract and grounding mechanisms

a novel approach that performs **natural language abductive reasoning via grounding-abstract chains** combining Linear Programming with Bayesian optimisation for science question answering

ExplanationLP answers and explains multiple- choice science questions via abductive natural lan- guage reasoning.

the task of answer- ing multiple-choice science questions is reformu- lated as the problem of **finding the candidate an- swer that is supported by the best explanation**

ExplanationLP constructs a **fact graph** where **each node is a fact**, and the **nodes and edges have a score** according to three prop- erties: relevance, cohesion and diversity.

an **optimal subgraph** is extracted using Linear Programming, whose role is to select the best sub-set of facts while preserving structural constraints imposed via grounding-abstract chains

**subgraphs’ global scores** computed by sum- ming up the nodes and edges scores are adopted to select the final answer. Since

Since the subgraph scores depend on the sum of nodes and edge scores, each property is multiplied by a learnable weight which is optimised via Bayesian Optimisation to obtain the best possible combination with the highest ac- curacy for answer selection. To

the first to combine a parameter optimisation method with Linear Programming for inference. The

**Relevance**: We promote the inclusion of highly relevant facts in the explanations by encouraging the selection of sentences with higher lexical rele- vance and semantic similarity with the hypothesis.

**Cohesion**: Explanations should be cohesive, implying that **grounding-abstract chains should re- main within the same context.** To achieve cohe- sion, we encourage a high degree of **overlaps be- tween different hops** (e.g. hypothesis-grounding, grounding-abstract, hypothesis-abstract) to prevent the inference chains from drifting away from the original context. 

**Diversity**: While maximizing relevance and co- hesion between different hops, we encourage **diver- sity between facts of the same typ**e (e.g. abstract- abstract, grounding-grounding) to address different parts of the hypothesis and promote completeness in the explanations.

positive impact of grounding-abstract mech- anisms on semantic drift.



#### Encoding Explanatory Knowledge for Zero-shot Science Question Answering - Zhou et al. 2021

N-XKT (Neural encod- ing based on eXplanatory Knowledge Trans- fer), a novel method for the automatic transfer of explanatory knowledge through neural en- coding mechanisms

able to improve accuracy and general- ization on science Question Answering (QA)

by leveraging facts from back- ground explanatory knowledge corpora, the N- XKT model shows a clear improvement on zero-shot QA

N- XKT can be fine-tuned on a target QA dataset, enabling faster convergence and more accurate results.

this work aims to explore the impact of explanatory knowledge on zero-shot generalisation

we argue that **explanation-centred**
**corpora** can serve as a resource to boost zero-shot capabilities on Question Answering tasks which demand deeper inference

we explore the adoption of **latent knowledge representations** for supporting generalisation on downstream QA tasks requiring multi-hop inference

**explanatory scientific**
**knowledge expressed in natural language can be transferred into neural network representations**, and subsequently used to achieve knowledge based in- ference on scientific QA tasks

this paper proposes a **unified approach that frames Question Answering as an explanatory knowledge reasoning problem**. The unification between the two tasks allows us to explore the adoption of pre-training strategies over explana- tory knowledge bases, and subsequently leverage the same paradigm to generalise on the Question Answering task

N-XKT, a neural mechanism for encoding and transferring explanatory knowl- edge for science QA.

the first work **tackling sci- ence QA tasks through the transfer of external explanatory knowledge via neural encoding mechanisms**

introduce the explanatory knowledge transfer task on explanation-centred knowl- edge bases

Scientific Question Answering has the distinctive
property of requiring the articulation of multi-hop and explanatory reasoning

the explanatory chains required to arrive at the correct answer typically operate at an **abstract level**, through the combina- tion of definitions and scientific laws

This characteristic makes the **gener-alisation process more challenging**, as the answer prediction model **needs to acquire the ability to per- form abstraction from the specific context in the question.**

au- tomatically transfer abstractive knowledge from ex- planatory facts into neural encoding representation for more accurate scientific QA, and for enabling zero-shot generalization. To

N-XKT (Neural encoding based on eXplanatory Knowledge Transfer) which encodes abstractive knowledge into neural representation to improve the effectiveness in both zero-shot QA task and fine-tuning based QA task.

evaluated adopting the fol- lowing training tasks:

1. Explanatory Knowledge Acquisition

2. Cloze-style Question Answering

we proposed a neural encoding mech- anism for explanatory knowledge acquisition and transfer, N-XKT.

The proposed model delivers better generalisation and accuracy for QA tasks that require multi-hop and explanatory inference

These results supports the hypothesis that pre- training tasks targeting abstract and explanatory knowledge acquisition can constitute and impor- tant direction to improve inference capabilities and generalization of state-of-the-art neural models



#### Identifying Supporting Facts for Multi-hop Question Answering with Document Graph Networks - Thayaparan et al. 2019



reading comprehension have resulted in models that surpass human performance when the answer is contained in a single, continuous passage of text. However, complex Question Answering (QA) typically requires **multi-hop reasoning** – i.e. the integra- tion of supporting facts from different sources, to infer the correct answer

Document Graph Net- work (DGN), a message passing architecture for the identification of supporting facts over a graph-structured representation of text

DGN obtains competitive results when compared to a reading comprehension baseline operating on raw text, confirming the relevance of struc- tured representations for supporting multi-hop reasoning.

**Question Answering** (QA) is the task of inferring the answer for a natural language question in a given knowledge source

Recent advances in deep learning have sparked interest in a specific type of QA emphasising Machine Comprehension (MC) aspects, where background knowledge is en- tirely expressed in form of unstructured text

State-of-the-art techniques for MC typically re-
trieve the answer from a continuous passage of text by adopting a combination of character and word-level models with various forms of attention mechanisms 

when it comes to answering complex questions on large document collections, it is un- likely that a single passage can provide sufficient evidence to support the answer. Complex QA typ- ically requires **multi-hop reasoning**, i.e. the abil- ity of combining multiple information fragments from different sources.

This paper explores the task of identifying sup-
porting facts for multi-hop QA over large collec- tions of documents where several passages act as distractors for the MC model

we hypothesise that graph-structured representations play a key role in reducing complexity and im- proving the ability to retrieve meaningful evidence for the answer.

identifying support-
ing facts in unstructured text is challenging as it requires capturing long-term dependencies to ex- clude irrelevant passages

On the other hand (Fig- ure 1.2), a graph-structured representation con- necting related documents simplifies the integra- tion of relevant facts by making them mutually reachable in few hops. We

transforming a text corpus in a global representation that links documents and sentences by means of mutual references

to identify supporting facts on undi-
rected graphs, we investigate the use of **message passing architectures with relational inductive bias**

We present the Document Graph Network (DGN), a specific type of Gated Graph Neural Network (GGNN) (Li et al., 2015) trained to identify supporting facts in the afore- mentioned structured representation

(DGN), a **message passing archi- tecture** designed to identify supporting facts for **multi-hop QA on graph-structured representations of documents.**

The advantage of using graph-structured repre- sentations lies in reducing the inference steps nec- essary to combine two or more supporting facts

we want to extract a representation that increases the probability of connecting relevant sentences with short paths in the graph. We ob- serve that multi-hop questions usually require rea- soning on two concepts/entities that are described in different, but interlinked documents. We put in practice this observation by connecting two docu- ments if they contain mentions to the same enti- ties.

we automatically ex-
tract a Document Graph DG encoding the back- ground knowledge expressed in a corpus of docu- ments (Step 1). This data and its graphical struc- ture is permanently stored into a database, ready to be loaded when it is required.

Document Graph Network (DGN),
a novel approach for selecting supporting facts in a multi-hop QA pipeline. The

The model operates over explicit relational knowledge, connecting docu- ments and sentences extracted from large text cor- pora. We adopt a pre-filtering step to limit the number of nodes and train a customised Graph Gated Neural Network directly on the extracted representation.

we highlight a way
to combine structured and distributional sentence representation models

#### Unification-based reconstruction of multi-hop explanations for science questions - Valentino et al. 2021

a novel framework for re- constructing multi-hop explanations in science Question Answering (QA).

While existing ap- proaches for multi-hop reasoning build expla- nations considering each question in isolation, we propose a method to l**everage explanatory patterns emerging in a corpus of scientific ex- planations.**

the framework ranks a set of atomic facts by integrating lexical rel- evance with the notion of unification power, estimated analysing explanations for similar questions in the corpus.

The need for explainability and a quantitative methodology for its evaluation have conducted to the creation of shared tasks on explanation recon- struction

Given a science question, **explanation reconstruction** consists in regenerating the gold ex- planation that supports the correct answer through the combination of a series of atomic facts.

Explana- tions for science questions are typically composed of two main parts: a **grounding part**, containing knowledge about concrete concepts in the ques- tion, and **a core scientific part**, including general scientific statements and laws.

constructing
long explanations is challenging due to **seman- tic drift** – i.e. the tendency of composing out-of- context inference chains as the number of hops increases 

While existing approaches build explanations con- sidering each question in isolation (Khashabi et al., 2018; Khot et al., 2017), we hypothesise that se- mantic drift can be tackled by leveraging **explana- tory patterns** emerging in clusters of similar ques- tions

In Science, a given statement is considered
**explanatory** to the extent it performs **unifica- tion** (Friedman, 1974; Kitcher, 1981, 1989), that is showing how a set of initially disconnected phe- nomena are the expression of the same regularity

Since the explanatory power of a given statement depends on the number of unified phenomena, highly explana- tory facts tend to create **unification patterns** – i.e. similar phenomena require similar explanations.

propose a method that leverages unification pat- terns for the reconstruction of multi-hop explana- tions

Recon- structing explanations for science questions can be reduced to a **multi-hop inference** problem, where multiple pieces of evidence have to be aggregated to arrive at the final answer

Aggregation methods based on lexical overlaps and explicit constraints suffer from **semantic drift** (Khashabi et al., 2019; Fried et al., 2015) – i.e. the tendency of composing spurious inference chains leading to wrong conclusions

One way to contain semantic drift is to **lever-**
**age common explanatory patterns in explanation- centred corpora** (Jansen et al., 2018). **Transform- ers** (Das et al., 2019; Chia et al., 2019) represent the state-of-the-art for explanation reconstruction in this setting (Jansen and Ustalov, 2019). However, these models require high computational resources that prevent their applicability to large corpora.

On the other hand, **approaches based on IR [Information Retrieval] techniques** are readily scalable. 

The approach described in this paper preserves the scalability of IR methods, obtaining, at the same time, performances competi- tive with Transformers. Thanks to this feature, the framework can be flexibly applied in combination with downstream question answering models

a novel framework for multi- hop explanation reconstruction based on **explana- tory unification**. 

The approach is competitive with state-of-the- art Transformers, yet being significantly faster and inherently scalable;

The unification-based mechanism supports the construction of complex and many hops explanations;

#### A Survey on Explainability in Machine Reading Comprehension - Thayaparan et al. 2020

Machine Reading Comprehension (MRC) has the long-standing goal of developing machines that can reason with natural language

a crucial requirement emerging in recent years is explainability (Miller, 2019), intended as the ability of a model to expose the underlying mechanisms adopted to arrive at the final answers

We refer to **explainability** as a specialisation of the higher level concept of **interpretability**.

In general, **interpretability** aims at developing tools to understand and investigate the behaviour of an AI system. This definition also includes tools that are external to a black-box model, as in the case of **post-hoc interpretability**

On the other hand, the goal of explainability is the design of **inherently interpretable** models, capable of performing transparent inference through the generation of an explanation for the final prediction (Miller,

In general, an **explanation** can be seen as an answer to a how question formulated as follows: “How did the model arrive at the conclusion c starting from the problem formulation p?”

In the context of MRC, the answer to this question can be addressed by exposing the internal reasoning mechanisms linking p to c. This goal can be achieved in two different ways

1. **Knowledge-based explanation**: exposing part of the relevant background knowledge connecting p and c in terms of supporting facts and/or inference rules;
2. **Operational explanation**: composing a set of atomic operations through the generation of a sym- bolic program, whose execution leads to the final answer c.

In **extractive MRC**, the reasoning required to solve the task is **derivable from the original problem formulation**. In other words, the correct decomposi- tion of the problem provides the necessary inference steps for the answer, and the role of the explanation is to fill an information gap at the **extensional level** – i.e. identifying the correct arguments for a set of predicates, via paraphrasing and coreference resolution. As a result, explanations for extractive MRC are often expressed in the form of supporting passages retrieved from the contextual paragraphs 

On the other hand, **abstractive MRC** tasks usually require going beyond the surface form of the problem with the inclusion of high level knowledge about **abstract concepts**., the expla- nation typically leverages the use of supporting definitions, including taxonomic relations and essential properties, to perform abstraction from the original context in search of high level rules and inference patterns

The ability to construct explanations in MRC is typically as- sociated with multi-hop reasoning. However, the nature and the structure of the inference can differ greatly according to the specific task. In

In extractive MRC (Yang et al., 2018; Welbl et al., 2018), **multi- hop reasoning** often consists in the identification of bridge entities, or in the extraction and comparison of information encoded in different passages

com- plete explanations for science questions require an average of 6 facts classified in three main explanatory roles: **grounding facts** and **lexical glues** have the function of connecting the specific concepts in the ques- tion with abstract conceptual categories, while **central facts** refer to high-level explanatory knowledge

. In general, the number of hops needed to construct the explanations is correlated with **semantic drift** – i.e. the tendency of composing spurious inference chains that lead to wrong conclu- sions

**Explicit Models**

typically adopt heuristics and hand-crafted constraints to encode high level hypotheses of explanatory relevance

* Linear Programming (LP)
* Weighting schemes with heuristics
* Pre-trained embeddings with heuristics; Pre-trained embeddings have the advantage of capturing se- mantic similarity, going beyond the lexical overlaps limitation imposed by the use of weighting schemes.

**Latent Models**

learn the notion of explanatory relevance implicitly through the use of machine learning techniques such as neural embeddings and neural language models

* Neural models for sentence selection: set of neural approaches proposed for the answer sentence selection problem; typically adopt deep learning architectures, such as RNN, CNN and Attention networks via strong or distant supervision.
  * Strongly supervised ap- proaches (Yu et al., 2014; Min et al., 2018; Gravina et al., 2018; Garg et al., 2019) are trained on gold supporting sentences. In
  * distantly supervised techniques indirectly learn to extract the support- ing sentence by training on the final answer. Attention mechanisms have been frequently used for distant supervision (Seo et al., 2016) to highlight the attended explanation sentence in the contextual passage.
* Transformers for multi-hop reasoning; Transformers-based architectures (Vaswani et al., 2017) have been successfully applied to learn explanatory relevance in both extractive and abstractive MRC tasks.
* Attention networks for multi-hop reasoning: , attention net- works have also been employed to extract relevant explanatory facts. However, attention networks are usually applied in combination with other neural modules.
* Language generation models

**Hybrid models**

adopt heuristics and hand-crafted constraints as a pre-processing step to impose an ex- plicit inductive bias for explanatory relevance. 

* Graph Networks: The relational inductive bias encoded in Graph Networks (Battaglia et al., 2018) provides a viable support for reasoning and learning over structured representations. This characteristic has been identified as particularly suitable for supporting facts selection in multi-hop MRC tasks. 
* Explicit inference chains for multi-hop reasoning: A subset of approaches has introduced end-to- end frameworks explicitly designed to emulate the step-by-step reasoning process involved in multi-hop MRC; The baseline approach proposed for Abductive Natural Language Inference (Bhagavatula et al., 2019) builds chains composed of hypotheses and observations, and encode them using transformers to identify the most plausible explanatory hypoth- esis. Similarly,

**Operational explanations** 

aim at providing interpretability by exposing the set of operations adopted to arrive at the final answer.

* Neuro-Symbolic models: Neuro-symbolic approaches combine neural models with symbolic pro- grams.
* Multi-hop question decomposition: The approaches in this category aim at breaking multi-hop ques- tions into single-hop queries that are simpler to solve.

Contrastive Explanations: while contrastive and conterfactual explanations are becoming central in Explainable AI (Miller, 2019), this type of explanations is still under-explored for MRC.