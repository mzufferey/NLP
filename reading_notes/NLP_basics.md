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



#### Learning to Explain: Datasets and Models for Identifying Valid Reasoning Chains in Multihop Question-Answering - Chains and Clark 2018

**multihop question-answering,** where multiple facts are needed to derive an answer.

a chain ofreasoning leading to an answer, can help a user assess an answer’s validity.

We are interested in questions where the decom-
position into subquestions - hence the explanation structure - is not evident from the question, but has to be found. For

For example, “Does a suit of armor conduct electricity?” might be answered (hence explained) by first identifying what material armor is made of, even though the question itself does not mention materials.

This contrasts with earlier multihop QA datasets, e.g., HotpotQA (Yang et al., 2018), where the explanation structure is evident in the question itself. For example, “What nation- ality was James Miller’s wife?” implies a chain of reasoning to first finds Miller’s wife, then her nationality. Such

For questions requiring inference, the focus of
this paper, an explanation is often taken as the chain of steps (typically sentences) leading to an answer

#### Dynamic Semantic Graph Construction and Reasoning for Explainable Multi-hop Science Question Answering - Xu et al. 2019

Abstract Meaning Representation (AMR) as semantic graph representation. Our

**Multi-hop QA** is one of the most challenging tasks that benefits from explainability as it mimics the human question answering setting, where multi- hop QA requires both the collection of information from large external knowledge resources and the aggregation of retrieved facts to answer complex natural language questions

Textual corpora contain rich and diverse evidence facts, which are ideal knowledge resources for multi-hop QA

To take advantages of both rich textual corpora
and explicit graph structure and make it compatible to all textual knowledge, we explore the usefulness of **Abstract Meaning Representation** (AMR) as a graph annotation to a textual fact

AMR (Banarescu et al., 2013) is a semantic formalism that represents the meaning of a sentence into a rooted, directed graph.

Unlike other semantic role labeling that only considers the relations between predicates and their arguments (Song et al., 2019), the aim ofAMR is to **capture every meaningful con- tent in high-level abstraction** while removing away inflections and function words in a sentence

AMR allows us to explore textual facts and simultaneously attributes them with explicit graph structure for explainable fact quality assessment and reasoning

we propose a novel framework
that incorporates AMR to make explainable knowl- edge retrieval and reasoning for multi-hop QA

The introduced AMR serves as a bridge that enables an explicit reasoning process over a graph structure among questions, answers and relevant facts. As

Unlike previous works on multi-hop QA that rely on existing KGs to find rela- tions among entities (Wang et al., 2020; Feng et al., 2020), our proposed AMR-SG is dynamically con- structed, which reveals intrinsic relations of facts and can naturally form any-hop connections

De- spite the success of pretrained model in most Natu- ral Language Processing (NLP) tasks, it performs poorly in multi-hop QA, where some information is missing to answer questions

#### Abductive commonsense reasoning - Bhagavatula et al. 2020

**Abductive reasoning** is inference to the most plausible explanation.

**Abductive reasoning** is inference to the most plausible explanation for incomplete observations

**abduction** is “the only logical operation which introduces any new ideas”, which contrasts with other types of inference such as entailment, that focuses on inferring only such information that is already provided in the premise

most previous work on abductive reasoning has focused on formal logic, which has proven to be too rigid to generalize to the full complexity of natural language.

we investigate the use of natural language as the representation medium, and probe deep neural models on language-based abductive reasoning.
, we propose Abductive Natural Language Inference (αNLI) and Abductive Natural Language Generation (αNLG) as two novel reasoning tasks in narrative contexts.

#### Multi-hop Question Answering via Reasoning Chains - Chen et al. 2021

**Multi-hop question answering** requires mod- els to gather information from different parts of a text to answer a question. Most

Most cur- rent approaches learn to address this task in an end-to-end way with neural networks, with- out maintaining an explicit representation of the reasoning process. We

We propose a method to extract a discrete reasoning chain over the text, which consists of a series of sentences leading to the answer. We

We then feed the ex- tracted chains to a BERT-based QA model (Devlin et al., 2018) to do final answer predic- tion. Critically,

, we do not rely on gold anno- tated chains or “supporting facts”: at training time, we derive pseudogold reasoning chains using heuristics based on named entity recog- nition and coreference resolution. Nor

Nor do we rely on these annotations at test time, as our model learns to extract chains from raw text alone.

modeling extraction sequentially is important, as is dealing with each candidate sentence in a context-aware way

models may need inductive bias if they are to solve this problem “correctly.”

we propose a step in this direction,
with a two-stage model that identifies intermediate reasoning chains and then separately determines the answer. A

A **reasoning chain** is a sequence of sentences that logically connect the question to a fact relevant (or partially relevant) to giving a rea- sonably supported answer. Figure

Extracting chains gives us a discrete intermediate output of the reasoning process, which can help us gauge our model’s behavior beyond just final task ac- curacy. Formally,

To find the right answer, we need to maintain uncertainty over this chain set, since the correct one may not immediately be evident, and for cer- tain types of questions, information across multi- ple chains may even be relevant. S

We use a search procedure leveraging coreference and **named en- tity recognition** (NER) to find a path from the start sentence to an end sentence through a graph of re- lated sentences.

We present
a method for extracting oracle reasoning chains for multi-hop reasoning tasks. These chains general- ize across multiple datasets and are comparable to human-annotated chains.

We present a model that learns from these chains at train time and at test time can produce a list of chains. Those chains could be used to gauge the behaviors of our model

A **reasoning chain** is a sequence of sen- tences that logically connect the question to a fact relevant to determining the answer. Two

Two adja- cent sentences in a reasoning chain should be intu- itively related: they should exhibit a shared entity



#### Is Multi-Hop Reasoning Really Explainable? Towards Benchmarking Reasoning Interpretability - Lv et al. 2021

Multi-hop reasoning has been widely studied in recent years to obtain more interpretable link prediction. However, we find in exper- iments that many paths given by these mod- els are actually unreasonable, while little work has been done on interpretability evaluation for them. In

we propose a unified framework to quantitatively evaluate the inter- pretability of multi-hop reasoning models so as to advance their development.

we define three metrics, including **path recall**, **local interpretability**, and **global interpretabil- ity** for evaluation, and design an approximate strategy to calculate these metrics using the in- terpretability scores of rules.

**Multi-hop reasoning for knowledge graphs** (KGs) has been extensively studied in recent years. It not only infers new knowledge but also provides reasoning paths that can explain the prediction re- sults and make the model trustable.

Most existing multi-hop reasoning models as-
sume that the output paths are reasonable and put much attention on the performance of link predic- tion.

In this paper, we propose a unified framework to automatically evaluate the interpretability of multi- hop reasoning models.

Multi-hop reasoning models can give interpretable paths while performing triple completion. Most of the existing multi-hop reasoning models are based on the reinforcement learning (RL) framework

Similar to multi-hop reasoning, **rule-based reason- ing** can also perform interpretable triple comple- tion, except that they give the corresponding rules instead of specific paths. Rule-based reasoning can be divided into two categories, namely, **neural- based models** and **rule mining models**. 

### Differentiable Convex Optimization for Explainable Multi-hop Natural Language Inference - Mohan

providing supporting evi- dence for Natural Language Inference

cast the problem as the construction of a **graph of natural lan- guage statements** that connects the premise to the hypothe- sis:

this graph is intended to serve as an **explanation for a valid inference**. In

**combinatorial optimization techniques** such as Integer Linear Programming (ILP) have been explored as a way to **encode explicit and controllable as- sumptions about the target graph**. 

such solvers provide a solution to the subgraph selection problem, they are often **limited by the use of explicit, predefined constraints** and **can- not be integrated as part of broader deep neural architectures**

In contrast, state-of-the-art **transformers** can learn from natu- ral language data and **implicitly encode complex constraints** for the inference. However, these models are intrinsically black boxes.

a novel framework named **∂- Explainer** (**Differentiable Explainer**) that aims at combining the best of both worlds: ∂-Explainer **integrates constrained optimization as part of a deep neural network via differen- tiable convex optimization**, allowing the fine-tuning of pre- trained transformers for downstream explainable NLP tasks.

we transform the constraints presented by TupleILP and integrate them with Transformer-based sentence embeddings for the task of explainable Science QA

**Constrained optimization solvers based on Integer Linear Programming** (ILP) have been proposed **as a method to ad- dress complex and knowledge-intensive natural language inference (NLI) tasks**

pro- vides a viable mechanism to **encode explicit and control- lable assumptions**, 

casting **multi-hop natural language inference  as an optimal subgraph selection problem.**

While delivering **explainability**, existing optimization solvers **cannot be integrated as part of a deep neural network** and are often **limited by the exclusive adoption of the constraints for inference**. 

prevents these methods from being optimized end-to-end on annotated cor- pora and

achieving performance and robustness comparable with deep learning counterparts

State-of-the-art models for natural language inference, in particular, are almost exclu- sively represented by **Transformers-based language models**, thanks to their **ability to transfer lin- guistic and semantic information to downstream tasks**

Transformers are typically regarded as **black-**
**box models**, posing concerns about the interpretability and transparency of their predictions

this paper proposes ∂-Explainer, **the first hybrid framework for multi-hop natu- ral language inference that combines constraint satisfaction layers with pre-trained neural representations**, enabling **end- to-end differentiability for optimization-based solvers**

**certain convex optimiza-**
**tion problems can be represented as individual layers** in larger end-to-end differentiable networks

these layers **can be adapted to encode constraints and dependen- cies between hidden states** that are hard to capture via stan- dard neural networks.

**convex optimization layers can be successfully integrated with Transformers** to achieve explainability and robustness in complex natural language inference problems.

we **transform the constraints into differentiable convex op- timization layers** and subsequently **integrate them with pre- trained sentence embeddings from Transformers** 

the performance of non-differentiable solvers can be improved by up to ≈ 10% when fine-tuned end-to-end while **still providing structured explanations in support of their inference**. In

A novel **differentiable framework** that **incorporates con- straints via convex optimization layers** into broader transformers-based architectures

the proposed framework **allows end-to-end differentiability on downstream tasks** for both explanation and answer selection, leading to a substantial improvement when compared to non- differentiable constraint-based solvers.

∂-Explainer is **more robust to distrac- tors when compared to Transformer-based models aug- mented with the same external evidence without the op- timization layer**

**ILP** has been employed to **model structural and semantic constraints to perform multi- hop natural language inference**. T

in line with previous works that have attempted to incor- porate optimization as a neural network layer.

we use the **differentiable convex optimization layers** pro- posed by Agrawal et al. (2019a). These layers provide a way to abstract away from the conic form, **letting users define convex optimization in natural syntax.**

The problem of **Multi-Hop Natural Language inference** can be stated as follows: Given a hypothesis h (each natural language sentences), we say that **we may infer h if there exists a subset of supporting facts in a knowledge base** {f1, f2, . . .} ⊆ F of true statements **which would allow a human being to deduce h from {f1, f2, . . .}**. We call this set of facts an **explanation** for h

we model
this as a **graph problem**: suppose that the **facts** in the knowl- edge base F **and the hypothesis h are the node**s of a prede- fined weighted graph G. We then wish to **find a connected subgraph of G that contains h, and is maximal with re- spect to the summed edge weights.**

**allows for the use of combinatorial optimization strate- gies** such as Integer Linear Programming (ILP),

the real challenge of ensuring that this produces convincing expla- nations lies in **assigning the edge weights** of the graph G (ideally **capturing a quantification of explanatory relevance**) and defining the **constraints** for the optimization problem

The novelty of our approach lies in the departure from the manually predefined edge weights of previous works (Khot, Sabharwal, and Clark 2017) to **weights that can be learnt dynamically from end-to-end multi-hop natural language in- ference examples**

Our approach proceeds in two strokes: 

1. incorporating an appropriate **dataset**, 
   * we **adapt a multi-hop question answering dataset into a multi-hop natural language inference dataset** by converting an example’s question (q) and the set of candidate answers C = {c1, c2, c3, . . . , cn} into hypotheses H = {h1, h2, h3, . . . , hn}
   * initialization of the knowledge graph, given the hypothe- ses H we adopt a retrieval model to select a list of candidate explanatory facts F = {f1, f2, f3, . . . , fk} to construct a weighted complete bipartite graph G = (H, F, E, W), where the weights Wik of each edge Eik denote how rele- vant a fact fk is with respect to the hypothesis hi.
2. designing an **end-to-end differen- tiable architecture** which simultaneously **solves the opti- mization problem and dynamically adjusts the graph edge weights** for better performance. 
   * departing from the standard ILP approaches, we
     adopt **differentiable convex optimization for the optimal sub- graph selection problem.**
   * we approximate and make differentiable the constraints presented in TupleILP 
     * TupleILP constructs a **semi-structured knowl- edge base using tuples** extracted via Open Information Ex- traction. 
     * It employs an **ILP model to perform inference over the extracted tuples** **taking into account the Subject- Predicate-Object structure** of the facts in the knowledge base. 
     * current state-of-the-art when considering the class of structured and integer linear pro- gramming solvers,
     * , the constraints of TupleILP are relatively easier to reproduce, providing more control to validate the contribution deriving

In previous work, the construction of the graph G re- quires **predetermined edge-weights** based on lexical over- laps or semantic similar- ity using sentence embeddings, on top of which combinatorial optimization strategies are performed separately.

we posit tha**t learning the graph weights dynamically as part of an end-to- end explainable natural language inference system** trained on examples that provide a gold-standard explanations for the correct answer will lead to more accurate and robust per- formance. To

To this end, the **optimization strategy should be differentiable and efficient**

TupleILP and similar Integer Linear Program- ming approaches present 2 key shortcomings that prevent achieving this goal: 

1. ILP formulation **inher- ently non-differentiable** as it results in a non-convex opti- mization problem -> cannot be integrated with deep neural networks and trained end-to-end
2. Integer Programming is known to be **NP-complete**; , as the size of the optimization problem in- creases, finding exact solutions becomes computation- ally intractable.; strong limitation for multi-hop natural language inference in general

we propose an **adaptation of the subgraph selection problem** so that the **edge-weighted representation of the graph G may also be optimized** during the training of the end-to-end reason- ing task.

we turn to **Semi-definite program- ming** (SDP) which is often used as a **convex approxima- tion of traditional NP-hard combinatorial graph optimiza- tion problems**, such

we lever- age the semi-definite relaxation of the following NP-hard subgraph selection problem

the semi-definite pro- gram relaxation can be solved by adopting the interior-point method (De

the first to employ SDP to solve a natural language processing task.

To demonstrate the impact of integrating a convex optimiza- tion layer into a broader end-to-end neural architecture, **∂- Explainer employs a Transformer-based sentence embed- ding model.**

**we incorporate a dif- ferentiable convex optimization layer with Sentence-BERT**

**SBERT** is adopted **to estimate the relevance between hypothesis and facts** during the construction of the base graph

SBERT as a bi-encoder architecture to minimize the computational overload and operate on large sentence graphs.

The **semantic relevance score from SBERT is com- plemented with a lexical relevance score** computed consid- ering the shared terms between hypotheses and facts. We

to adopt **differentiable convex optimization layers**, the constraints **should be defined following the Disciplined Parameterized Programming (DPP) formalism** (Agrawal et al. 2019a), providing a set of conventions when construct- ing convex optimization problems

* DPP consists of **func- tions (or atoms) with a known curvature** (affine, convex or concave) and **per-argument monotonicities.** 
* also consists of **Parameters which are symbolic constants** with an unknown numerical value assigned during the solver run. 

In addition to the aforementioned constraints and semidefinite constraints specified in Equation 3, we adopt part of the **constraints from TupleILP** (

The output from the DCX layer returns **the solved edge ad- jacency matrix** ˆE with values between 0 and 1.

We interpret the **diagonal values of**
**Eˆ be the probability of the specific**
**node to be part of the selected subgraph.** The

The final step is to optimize the sum of the **cross-entropy loss** lc **between the se- lected answer and correct answer** hans, as well as the **binary cross entropy loss** lb **between the selected explanatory facts and true explanatory facts** Fexp

open new lines of research on the integration of neural networks and constrained optimization, leading to more controllable, transparent and explainable NLP models.



### GLOSSARY FOR Differentiable Convex Optimization for Explainable Multi-hop Natural Language Inference - Mohan

differentiable optimization problems = problems whose solutions can be backpropagated through

ng, combinatorial optimization techniques such as Integer Linear Programming (ILP)

differen- tiable convex optimization

optimal subgraph selection problem

incorporates con- straints via convex optimization layers

robust to distrac- tors

Constraint-Based NLI Solvers:

building semi-structured representations using Open Infor- mation Extraction

These layers provide a way to abstract away from the conic form, letting users define convex optimization in natural syntax

knowledge graph

retrieval model

a semi-structured knowl- edge base using tuples extracted via Open Information Ex- traction. It

subgraph selection problem	

we turn to Semi-definite program- ming (SDP) which is often used as a convex approxima- tion of traditional NP-hard combinatorial graph optimiza- tion problems,

positive semidefinite matrices satisfying

The optimal solution matrix
Eˆ is selected from the cone
of positive semi-definite matrices

the semi-definite pro- gram relaxation can be solved by adopting the interior-point method

h Sentence-BERT (SBERT)

to adopt differentiable convex optimization layers, the constraints should be defined following the Disciplined Parameterized Programming (DPP) formalism (Agrawal et al. 2019a), providing a set of conventions when construct- ing convex optimization problems. DPP consists of func- tions (or atoms) with a known curvature (affine, convex or concave) and per-argument monotonicities. In addition to these, DPP also consists of Parameters which are symbolic constants with an unknown numerical value assigned during the solver run. W

SPO tuples

Open Information Extraction model (Stanovsky

DCX layer returns



​        **Cone programs** are optimization problems that minimize a linear functional over        the intersection of an affine subspace and a convex cone 

 Any convex constraint can be represented as a conic constraint, so not every cone program is efficiently solvable. Even so, many commonly occurring cones give rise to tractable optimization problems, making cone programming a useful unifying framework. 

In linear algebra, a **convex cone** is a subset of a vector space over an ordered field that is closed under linear combinations with positive coefficients. 

A subset C of a vector space V over an ordered field F is a **cone** (or sometimes called a linear cone) if for each x in C and positive scalar α in F, the product αx is in C

A cone *C* is a **convex cone** if *αx* + *βy* belongs to *C*, for any positive scalars *α*, *β*, and any *x*, *y* in *C*.

A cone *C* is **convex** if and only if *C* + *C* ⊆ *C*.

Cone programming is a broad generalization of linear programming. 

Cone programming is a natural abstraction of semidefinite programming in which
we can conveniently develop some basic theory, most notably semidefinite pro-
gramming duality. 

 Conic combinations are similar to affine combinations but there is a difference in the constraint. ([tds](https://towardsdatascience.com/optimization-algorithms-for-machine-learning-d98d0feef53e))

Un c ˆone est donc une union de demi-droites ferm ́ees issues de l’origine

!!! à lire - sur l'optimisation: https://cel.archives-ouvertes.fr/cel-00356686/document

Un programme conique sur un cˆone convexe r ́egulier K ⊂Rn est
un probl`eme d’optimisation de la forme
minx ∈K 〈c ,x 〉: Ax = b

**tout probl`eme d’optimisation convexe peut ˆetre reformul ́e comme**
**programme conique**

l’ensemble faisable d’un
programme conique est
l’intersection du cˆone K
avec un sous-espace affine

(source: https://membres-ljk.imag.fr/Roland.Hildebrand/hildebrand_bourget.pdf)

Cˆones: Rˆole essentiel dans la formulation des contraintes in ́egalit ́es.

Un cˆone est donc une union de demi-droites ferm ́ees issues de
l’origine.

* Les espaces vectoriels sont  ́evidemment des cˆones
* Un cˆone est dit saillant si C ∩(−C ) = {0}
* Un espace vectoriel n’est pas un cˆone saillant.
* Un espace vectoriel est un cˆone convexe.

(source: http://cermics.enpc.fr/~jpc/optimisation-files/Ponts-slides.pdf)

**Convex programming**  is a subclass of  nonlinear programming (NLP) that uni- 
fies  and generalizes  least  squares (LS), linear programming  (LP), and convex 
quadratic programming  (QP). 

we  introduce a new modeling methodology called  **disciplined** 
**convex programming**. As the term  "disciplined"  suggests, the methodology im- 
poses a set of  conventions that one must follow when constructing convex pro- 
grams. 

s. The conventions do not limit generality; 
but  they  do  allow  much  of  the manipulation  and transformation  required  to 
analyze and  solve convex  programs t o   be  automated

 new way  to define a **function** in a modeling 
framework:  **as the solution  of  a  disciplined  convex  program**.  

We  call  such  a 
definition a **graph implementation**, so named  because  it exploits the properties 
of epigraphs and  hypographs  of  convex  and  concave  functions, respectively

The benefits of graph implementations  to are significant, because they provide 
a  means t o   support  nondifferentiable  functions  without  the loss of  reliability 
or performance  typically  associated  with  them

(source: https://web.stanford.edu/~boyd/papers/pdf/disc_cvx_prog.pdf)

see Convex optimization book: https://web.stanford.edu/~boyd/cvxbook/ (book and slides available)

https://www.cvxpy.org/tutorial/dcp/index.html

---------------------



### Differentiable Convex Optimization Layers - Agrawal et al. 2019

embed differentiable optimization problems (that is, problems whose solutions can be backpropagated through) as layers within deep learning architectures.

provides a useful inductive bias for certain problems, but

we propose an **approach to differ- entiating through disciplined convex programs**, a subclass of convex optimization problems used by domain-specific languages (DSLs) for convex optimization

we introduce **disciplined parametrized programming**, a subset of disciplined convex programming

every disciplined parametrized program can be represented as the composition of 

* a solver = an affine map from parameters to problem data
* an affine map from the solver’s solution to a solution of the original problem

 (a new form we refer to as affine-solver-affine form).

efficiently differentiate through each of these components, allowing for end-to-end analytical differentiation through the entire convex program. We

convex optimization problems = functions mapping problem data to solutions

convex optimization layers can provide useful inductive bias in end-to-end models, their adoption has been slowed by how difficult they are to use. 

Existing layers (e.g., [6, 1]) require users to transform their problems into rigid canonical forms by hand

. **Domain-specific languages** (DSLs) for convex optimization 

* abstract away the process of converting problems to canonical forms, 
* letting users specify problems in a natural syntax;
* programs are then lowered to canonical forms and supplied to numerical solvers behind-the-scenes

* enable rapid prototyping and make convex optimization accessible to scientists and engineers not experts in optimization.

this paper: **do what DSLs have done for convex optimization, but for differentiable convex optimization layers**

 we show how to efficiently differentiate through **disciplined convex programs** = large class of convex optimization problems that can be parsed and solved by most DSLs for convex optimization

we introduce **disciplined parametrized programming** (DPP), **a grammar for producing parametrized disciplined convex programs**. 

Given **a program produced by DPP**, we show how to **obtain an affine map from parameters to problem data**, and **an affine map from a solution of the canonicalized problem to a solution of the original problem**. 

this representation of a problem = **affine-solver-affine (ASA) form** = the composition of an affine map from parameters to problem data, a solver, and an affine map to retrieve a solution 


We introduce 

* **DPP = a new grammar for parametrized convex optimization problems**
* **ASA form = ensures that the mapping from problem parameters to problem data is affin**



**DSLs for convex optimization** 

* allow users to specify convex optimization problems in a natural way that follows the math
* at the foundation: ruleset from convex analysis known as **disciplined convex programming**
  * **disciplined convex program** = a mathematical program written using DCP 
    * all such programs are **convex**. 
    * can be **canonicalized to cone programs** by expanding each nonlinear function into its graph implementation 
  *  **DPP can be seen as a subset of DCP** that mildly restricts the way parameters (symbolic constants) can be used



**Differentiation of optimization problems** 

* convex optimization problems do not in general admit closed-form solutions
* nonetheless possible to **differentiate through convex optimization problems by implicitly differentiating their optimality conditions** (when certain regularity conditions are satisfied)

* methods were developed to **differentiate through convex cone programs** 
  * general methods since **every convex program can be cast as a cone program**

The software released requires users to express their problems in conic form. Expressing a convex optimization problem in conic form requires a working knowledge of convex analysis. 

**our work abstracts away conic form, letting the user differentiate through high-level descriptions of convex optimization problems**; we canonicalize these descriptions to cone programs on the user’s behalf. This makes it possible to rapidly experiment with new families of differentiable programs, induced by different kinds of convex optimization problems.

Because **we differentiate through a cone program by implicitly differentiating its solution map**, our method can be paired with any algorithm for solving convex cone programs

A parametrized **convex optimization problem** can be viewed as a (possibly multi-valued) function that maps a parameter to solutions.

**Disciplined convex programming**

*  is **a grammar for constructing convex optimization prob- lems** 
* consists of 
  * **functions**, or atoms, 
    * atom = a function with known curvature (affine, convex, or concave) and per-argument monotonicities
  * and a single rule for composing them.

* Every disciplined convex program is a convex optimization problem, but the converse is not true
  * not a limitation in practice, because atom libraries are extensible (i.e., the class corresponding to DCP is parametrized by which atoms are implemented). 

**Cone programs.**

*  A (convex) cone program is an optimization problem 
* Our method for differentiating through disciplined convex programs requires calling a **solver** (an algorithm for solving an optimization problem) in the forward pass. We

* focus on the special case in which the solver is a **conic solver**. 
  * A conic solver targets convex cone programs,

**Disciplined parametrized programming** (DPP) is a grammar for producing parametrized disciplined convex programs from a set of functions, or atoms, with known curvature (constant, affine, convex, or concave) and per-argument monotonicities. 

A program produced using DPP is called **a disciplined parametrized program**.

Like DCP, DPP is based on the well-known composition theorem for convex functions, and it guarantees that every function appearing in a disciplined parametrized program is affine, convex, or concave.

Unlike DCP, DPP also guarantees that the produced program can be reduced to ASA form

A disciplined parametrized program is an optimization problem

* An **expression** can be thought of as a tree, where the nodes are atoms and the leaves are variables, constants, or parameters. 
* A **parameter** is a symbolic constant with known properties such as sign but unknown numeric value. 
* An expression is said to be **parameter-affine** if it does not have variables among its leaves and is affine in its parameters; 
* an expression is **parameter-free** if it is not parametrized, and variable-free if it does not have variables.

Every DPP program is also DCP, but the converse is not true.

DPP generates programs reducible to ASA form by introducing two restrictions on expressions involving parameters

The **canonicalization** of a disciplined parametrized program to ASA form is similar to the canoni- calization of a disciplined convex program to a cone program

The full canonicalization procedure (which includes expanding graph implementations) only runs the first time the problem is canonicalized. When

### Answering Complex Questions Using Open Information Extraction Tushar - Khot et al. 2017

Open Information Extraction (Open IE) provides a way to generate semi-structured knowledge for QA, but to date such knowledge has only been used to answer simple questions with retrieval- based methods.

presenting **a method for reasoning with Open IE knowledge,** allowing more complex questions to be handled.

Using a recently proposed **support graph optimiza- tion** framework for QA, we develop **a new inference model for Open IE**, in particu- lar one that can work effectively with mul- tiple short facts, noise, and the relational structure of tuples. Our

r, these KBs are expensive to build and typically domain-specific

Automatically con- structed open vocabulary (subject; predicate; ob- ject) style tuples have broader coverage, but have only been used for simple questions where a single tuple suffices (Fader

develop a **QA system that can perform reasoning with Open IE tuples** for complex multiple-choice questions that **require tuples from multiple sen- tences**

* Such a system can answer complex ques- tions in resource-poor domains where curated knowledge is unavailable.
  * Elementary-level sci- ence exams is one such domain, requiring com- plex reasoning (Clark,
    * lack of a large-scale structured KB -> either rely on shallow reasoning with large text corpora or deeper, structured reasoning with a small amount of automatically acquired or manually curated knowledge

Which object in our solar system reflects
light and is a satellite that orbits around one planet? (A) Earth (B) Mercury (C) the Sun (D) the Moon

-> A natural way to answer it is by combining facts such as (Moon; is; in the solar system), (Moon; reflects; light), (Moon; is; satellite), and (Moon; orbits; around one planet).

ex for such reasoning: TableILP

TABLEILP 

* treats **QA as a search for an optimal subgraph** that **connects terms in the question and answer via rows in a set of curated tables**
* solves the optimization problem using **Integer Linear Programming** (ILP)

We similarly want to **search for an optimal subgraph** but **large, automatically extracted tuple KB** makes the reasoning context different on three fronts

* unlike reasoning with tables, **chaining tuples is less important and reliable** as join rules aren’t available; 
* **conjunctive evidence becomes paramount**, as, unlike a long table row, **a single tuple is less likely to cover the entire question**; and
* unlike table rows, **tuples are noisy**, making **combining redundant evidence essen- tial**. 

-> table-knowledge centered inference model isn’t the best fit for noisy tuples

**a new ILP-based model of inference with tuples**, im- plemented in a reasoner called **TUPLEINF**. 

demonstrates for the first time how **Open IE based QA can be extended from simple lookup questions to an effective system for complex questions**

The work most related to TUPLEINF is the aforementioned **TABLEILP** solver. 

* focuses on **building inference chains using manually ually defined join rules for a small set of curated tables**.
* can also use open vocabulary tu- ples but efficacy limited by the **difficulty of defining reliable join rules for such tuples**. 
* **each row in some complex curated tables covers all relevant contex- tual information** (e.g., each row of the adaptation table contains (animal, adaptation, challenge, ex- planation)), whereas **recovering such information requires combining multiple Open IE tuples**

. We define a tuple as (subject; predicate; objects) with zero or more objects. We

, we use the corresponding training questions Qtr to retrieve domain-relevant sentences from S.

Given a multiple-choice question qa we select the most relevant tuples from T and S as follows

* from tupleKB
  * use an in-
    verted index to find the 1,000 tuples that have the most overlapping tokens with question tokens tok(qa).
  *  filter out any tuples that over- lap only with tok(q) as they do not support any answer. 
  * compute the normalized TF-IDF score
* on the fly
  * To handle ques-
    tions from new domains not covered by the train- ing set, we extract additional tuples on the fl
  * run Open IE on these sentences and re-score the resulting tuples using the Jaccard score7

Similar to TABLEILP, we view **the QA task as**
**searching for a graph that best connects the terms in the question (qterms) with an answer choice via the knowledge**;

The qterms, answer choices, and tuples fields
form the set of possible **vertices**, V, of the support graph

Edges connecting qterms to tuple fields and tuple fields to answer choices form the set of pos- sible **edges**,

We define the desired behavior of an optimal support graph via an ILP model

*Objective function*

Similar to TABLEILP, we score the support graph based on the weight of the active nodes and edges. 

* Each edge e(t, h) is **weighted based on a word- overlap score**
  * TABLEILP used Word- Net (Miller, 1995) paths to compute the weight
  * results in unreliable scores when faced with longer phrases found in Open IE tuples
  * improve the scoring of qterms in our ILP objective to focus on important terms. 
  * the later terms in a ques- tion tend to provide the most critical information, we scale qterm coefficients based on their position
  * qterms that appear in almost all of the se- lected tuples tend not to be discriminative as any tuple would support such a qterm. Hence we scale the coefficients by the inverse frequency of the to- kens in the selected tuples

*Constraints*

 define active vertices and edges using ILP constraints: an active edge must connect two ac- tive vertices and an active vertex must have at least one active edge. 

To avoid spurious tuples that only connect with the question (or choice) or ignore the relation being expressed in the tuple, we add constraints that require each tuple to connect a qterm with an answer choice

Since an Open IE tu- ple expresses a fact about the tuple’s subject, we require the subject to be active in the support graph. To

, we also add an ordering constraint 

For reliable multi-hop reasoning using OpenIE tu- ples, we can add inter-tuple connections to the support graph search, controlled by a small num- ber of rules over the OpenIE predicates.

a new QA system, TUPLEINF, that can reason over a large, potentially noisy tuple KB to answer complex questions

### Question Answering via Integer Programming over Semi-Structured Knowledge  - Khashabi et al. 2016

We propose a structured in- ference system for this task, formulated as an **In- teger Linear Program (ILP), that answers natural language questions using a semi-structured knowl- edge base** derived from text, including questions requiring multi-step inference and a combination of multiple facts.

Information Retrieval (IR) systems work under the as- sumption that answers to many questions of interest are of- ten explicitly stated somewhere [

statistical correlation based methods, such as those using Pointwise Mutual Information or PMI [Church and Hanks, 1989], work under the assumption that many questions can be answered by looking for words that tend to co-occur with the question words in a large corpus

both not suitable for questions requiring reasoning

TableILP 

* searches for the best support graph (chains of reasoning) connecting the question to an answer

* Constraints on the graph define what con- stitutes valid support and how to score it

We would like a QA system that, even if the answer is not ex- plicitly stated in a document, can combine facts to answer a question

wewould like the system to be **robust under simple perturbations**, such as changing New York to New Zealand (in the southern hemisphere) or changing an incorrect answer option to an irrelevant word such as “last” that happens to have high co-occurrence with the question text

we propose a **structured reasoning system**,
called **TableILP**, that **operates over a semi-structured knowl- edge base** derived from text and **answers questions by chain- ing** multiple pieces of information and **combining parallel evidence**

**knowledge base consists of tables,** each of which is a collection of instances of an n-ary relation defined over natural language phrases

treats **lex- ical constituents of the question Q, as well as cells of poten- tially relevant tables T, as nodes in a large graph** GQ,T, and attempts to find a subgraph G of GQ,T that “best” supports an answer option. The

notion of best support is captured via a number of **structural and semantic constraints and prefer- ences,** which are conveniently **expressed in the Integer Linear Programming (ILP) formalism.**

ILP optimization engine called SCIP to determine the best supported answer for Q

TableILP benefits from the table structure, by comparing it with an IR system using the same knowledge (the table rows) but ex- pressed as simple sentences; TableILP

our approach is robust to a simple perturbation of incorrect answer options

many science questions have answers that are not explic- itly stated in text, and instead require combining informa- tion together

there are AI systems for formal scientific reasoning they require questions to be posed in logic or restricted English. 

Our goal here is a system that operates be- tween these two extremes, able to **combine information while still operating with natural language**

semi-structured knowledge represented in the form of n-ary predicates over natural language text [Clark

* a k-column table in the knowledge base is a predicate r(x1, x2, . . . , xk) over strings, where each string is a (typically short) natural language phrase.
* The column head- ers capture the table schema, akin to a relational database. 
* Each row in the table corresponds to an instance of this predicate

Since ta- ble content is specified in natural language, the same entity is often represented differently in different tables, posing an additional inference challenge

Tables were constructed using a mixture of manual and semi- automatic techniques. First,

question answering as the task of **pairing the ques- tion with an answer such that this pair has the best support in the knowledge base**, measured in terms of the **strength of a “support graph”**

an **edge** denotes (soft) equality between a ques-
tion or answer node and a table node, or between two table nodes.

To **account for lexical variability** (e.g., that tool and in- strument are essentially equivalent) and **generalization** (e.g., that a dog is an animal), we replace string equality with a **phrase-level entailment or similarity function**

we would like **the support graph for an answer**
**option to be connected, and to include nodes from the ques- tion, the answer option, and at least one table**.

Since each table row represents a coherent piece of information but cells within a row do not have any edges in GQ,T (the same holds also for cells and the corresponding column headers), we use the notion of an **augmented subgraph** to capture the underly- ing table structure.

A **support graph** thus **connects the question constituents to**
**a unique answer option through table cells and (optionally) table headers** corresponding to the aligned cells. 

**A given question and tables give rise to a large number of possible support graphs, and the role of the inference process will be to choose the “best” one** under a notion of desirable support graphs developed next. 

We do this through a number of **addi- tional structural and semantic properties; the more properties the support graph satisfies, the more desirable it is.**

We model the above **support graph search** for QA as an **ILP**
**optimization problem**, = ., as **maximizing a linear objective function over a finite set of variables, subject to a set of linear inequality constraints**

the **ILP objective and constraints** aren’t
tied to the particular domain of evaluation; they **represent general properties that capture what constitutes a well sup- ported answer for a given question.**

All core vari- ables in the ILP model are binary

For each element, the model has a unary variable capturing whether this element is part of the support graph G, i.e., it is “active”. For

These unary and pairwise variables are then used to define various types of constraints and preferences, a

in practice we do not create
all possible pairwise variables. Instead we choose the pairs alignment score w(e) exceeds a pre-set threshold. For

The objective function is a weighted linear sum over all
variables instantiated for a given question answering prob- lem.

**Constraints** are a significant part of our model, **used for imposing the desired behavior on the support graph**. Some of them

* basic lookup
* parallel evidence
* Evidence Chaining
* Semantic Relation Matching
  

We treat QA as a subgraph selection problem and then formulate this as an ILP optimization.

, this formulation allows multiple, semi-formally expressed facts to be combined to answer questions, a capability outside the scope of IR-based QA systems.

### OptNet: Differentiable Optimization as a Layer in Neural Networks Brandon - Amos and Kolter 2017

OptNet, a network architec- ture that **integrates optimization problems as individual layers** in larger end-to-end train- able deep networks

These layers **encode con- straints and complex dependencies between the hidden states** that traditional convolutional and fully-connected layers often cannot capture

ability of our architecture to learn hard constraints better than other neural architec- tures

how to **treat exact, constrained optimization as an individual layer** within a deep learn- ing architecture. Unlike

Unlike traditional feedforward networks, where the output of each layer is a relatively simple (though non-linear) function of the previous layer

 our optimization framework **allows for individual layers to capture much richer behavior**, expressing complex operations that in total can reduce the overall depth of the network while pre- serving richness of representation.

**the output of the i + 1th layer in a net- work is the solution to a constrained optimization problem based upon previous layers**

This framework naturally en- compasses a wide variety of inference problems expressed within a neural network, allowing for the potential of much richer end-to-end training for complex tasks that require such inference procedures

these parameters can depend in any differentiable way on the previous layer zi, and which can eventually be optimized just like any other weights in a neural network. 	

These layers can be learned by taking the gradients of some loss function with respect to the parameters.

to the make the approach practical for larger net- works, we develop a custom solver which can simultane- ously solve multiple small QPs in batch form. 

. One cru- cial algorithmic insight in the solver is that by using a specific factorization of the primal-dual interior point up- date, we can obtain a backward pass over the optimiza- tion layer virtually “for free” (i.e., requiring no additional factorization once the optimization problem itself has been solved). Together,

we do not unroll an optimization procedure but instead use argmin differentiation as described in the next section.

A notable dif- ferent from other work within ML that we are aware of, is that we analytically differentiate through inequality as well as just equality constraints, but differentiating the comple- mentarity conditions; this

in the most general form, an OptNet layer can be any optimization problem, in

In the neural network setting, the optimal solution (or more generally, a subset ofthe optimal solution) of this optimization prob- lems becomes the output of our layer, denoted zi+1, and any of the problem data Q, q, A, b, G, h can depend on the value of the previous layer zi. The

The forward pass in our Opt- Net architecture thus involves simply setting up and finding the solution to this optimization problem.

Training deep architectures, however, requires that we not just have a forward pass in our network but also a back- ward pass. This requires that we compute the derivative of the solution to the QP with respect to its input parameters

To ob- tain these derivatives, we differentiate the KKT conditions (sufficient and necessary conditions for optimality) of (2) at a solution to the problem using techniques from matrix differential calculus (Magnus

today’s state-of-the-art QP solvers like Gurobi and CPLEX do not have the capability of solving multi- ple optimization problems on the GPU in parallel across the entire minibatch. This makes larger OptNet layers be- come quickly intractable compared to a fully-connected layer with the same number of parameters.
To

A key point of the particular form of primal-dual interior point method that we employ is that it is possible to com- pute the backward pass gradients “for free” after solving the original QP, without an additional matrix factorization or solve. Specifically,

all
the backward pass gradients can be computed using the factored KKT matrix at the solution. Crucially,

while the OptNet layers can be trained just as any neural network layer, since they are a new cre- ation and since they have manifolds in the parameter space which have no effect on the resulting solution (e.g., scaling the rows of a constraint matrix and its right hand side does not change the optimization problem), there is admittedly more tuning required to get these to work.



### BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding - Devlin et al. 2019

a new language representa- tion model called BERT, which stands for Bidirectional Encoder Representations from Transformers

. Unlike recent language repre- sentation models BERT is designed to **pre- train deep bidirectional representations from unlabeled text** by jointly conditioning on both left and right context in all layers. 

As a re- sult, the pre-trained BERT model **can be fine- tuned with just one additional output layer** to create state-of-the-art models for a wide range of tasks

2 existing strategies for **apply-**
**ing pre-trained language representations** to down- stream tasks: 

1. **feature-based** (e.g. ELMO)
   * uses task-specific architectures that **include the pre-trained representations as addi- tional features**
2. **fine-tuning** (e.g. OpenAI GPT)
   * introduces minimal task-specific parameters, and is trained on the downstream tasks by simply **fine-tuning all pre- trained parameters**

The two approaches **share the same objective function during pre-training**, where they use unidirectional language models to learn general language representations.

standard language models are unidirectional, and this limits the choice of archi- tectures that can be used during pre-training

OpenAI **GPT**:  left-to- right architecture, where every token **can only at- tend to previous tokens** in the self-attention layers of the Transformer

BERT alleviates the previously mentioned unidi- rectionality constraint by using a **“masked lan- guage model” (MLM) pre-training objective**, in- spired by the Cloze task

* randomly masks some of the tokens from the input
*  the objective is to predict the original vocabulary id of the masked word based only on its context

the MLM ob- jective enables the representation to fuse the left and the right context, which allows us to pre- train a deep bidirectional Transformer.

we also use a **“next sentence prediction” task** that jointly pre- trains text-pair representations. The

BERT uses **masked language models to enable pre- trained deep bidirectional representations**

pre-trained representations reduce the need for many heavily-engineered task- specific architectures. BERT

BERT is the first **fine- tuning based representation model** that achieves state-of-the-art performance on a large suite of **sentence-level <u>and</u> token-level tasks,**

*Unsupervised Feature-based Approaches Learning*

Learning widely applicable representations of words has

**ELMo** and its predecessor  generalize traditional word embedding re- search along a different dimension. They extract **context-sensitive features from a left-to-right and a right-to-left language model**; The contextual rep- resentation of each token is the **concatenation of the left-to-right and right-to-left representations.**

*Unsupervised Fine-tuning Approaches*

sentence or document encoders
which produce contextual token representations have been pre-trained from unlabeled text and fine-tuned for a supervised downstream task

The advantage of these approaches is that few parameters need to be learned from scratch. At

2 steps in our framework: 

1. pre-training and 
   * the model is trained on unlabeled data over different pre-training tasks.
2. fine-tuning
   * BERT model is first initialized with the pre-trained parameters, and all of the param- eters are fine-tuned using labeled data from the downstream tasks.
   * Each downstream task has sep- arate fine-tuned models, even though they are ini- tialized with the same pre-trained parameters

A distinctive feature of BERT is its unified ar- chitecture across different tasks

minimal difference between the pre-trained architec- ture and the final downstream architecture

BERT’s model architec- ture is a **multi-layer bidirectional Transformer en- coder** 

the BERT Transformer uses **bidirectional self-attention**, while the GPT Trans- former uses constrained self-attention where every token can only attend to context to its left

our input representation is able to unambiguously represent both a single sentence and a pair of sentences (e.g., ? Question, Answer ?) in one token sequence

a “**sentence**” can be an arbi- trary span of contiguous text, rather than an actual linguistic sentence.

A “**sequence**” refers to the in- put token sequence to BERT, which may be a sin- gle sentence or two sentences packed together.

* **WordPiece embeddings** with a 30,000 token vocabulary. The

* The first token of every sequence is always a special **clas- sification token** ([CLS]).
  * The final hidden state corresponding to this token is used as the ag- gregate sequence representation for classification tasks. Sentence

* Sentence pairs are packed together into a single sequence. differentiate the sentences in 2 ways
  * separate them with a special token ([SEP])
  *  add a learned embed- ding to every token indicating whether it belongs to sentence A or sentence B. 

For a given token, its **input representation is constructed by summing the corresponding token, segment, and position embeddings**. 

we do not use traditional left-to-right or right-to-left language models to pre-train BERT. Instead, we **pre-train BERT using two unsuper- vised tasks**

1. masked LM (Cloze task)
   * standard conditional language models can only be trained left-to-right or right-to-left, since bidirec- tional conditioning would allow each word to in- directly “see itself”, and the model could trivially predict the target word in a multi-layered context.
   * simply mask some percentage of the input tokens at random, and then predict those masked tokens. 
   * , the final hidden vectors corresponding to the mask tokens are fed into an output softmax over the vocabulary
   * mask 15% of all WordPiece to- kens in each sequence at random
   * only predict the masked words rather than recon- structing the entire input
   * downside: we are creating a mismatch between pre-training and fine-tuning, since the [MASK] token does not ap- pear during fine-tuning; To mitigate this, not always replace “masked” words with the ac- tual [MASK] token
     * 15% of the token positions at random for prediction. If the i-th token is chosen, we replace the i-th token with 
       * (1) the [MASK] token 80% of the time 
       * (2) a random token 10% of the time 
       * (3) the unchanged i-th token 10% of the time.
   * Then, Ti will be used to predict the original token with cross entropy loss. We
2. Next Sentence Prediction (NSP)
   * Many important downstream tasks are based on understanding the rela- tionship between two sentences, which is not di- rectly captured by language modeling
   * train a model that understands sentence rela- tionships,:  pre-train for a binarized next sen- tence prediction task that can be trivially gener- ated from any monolingual corpus. 
   * when choosing the sentences A and B for each pre- training example, 
     * 50% of the time B is the actual next sentence that follows A (labeled as IsNext)
     * 50% of the time it is a random sentence from the corpus (labeled as NotNext)

in prior work, only sentence embeddings are transferred to down-stream tasks, where **BERT transfers all pa- rameters to initialize end-task model parameters**



Fine-tuning is straightforward since the self- attention mechanism in the Transformer al- lows BERT to model many downstream tasks—

For each task, we simply plug in the task-
specific inputs and outputs into BERT and fine- tune all the parameters end-to-end. 

At the in- put, sentence A and sentence B from pre-training are analogous to e.g. question-passage pairs in question answering,

At the output, **the token rep- resentations are fed into an output layer for token- level tasks**, such as sequence tagging or question answering, and **the [CLS] representation is fed into an output layer for classification**, such as en- tailment or sentiment analysi





Figure 1: Overall pre-training and fine-tuning procedures for BERT. Apart from output layers, the **same architec- tures are used in both pre-training and fine-tuning**. The **same pre-trained model parameters are used to initialize models for different down-stream tasks**. **During fine-tuning, all parameters are fine-tuned**. 



### BioBERT: a pre-trained biomedical language representation model for biomedical text mining - Lee et al. 2020

we investigate how the recently introduced pre-trained language model BERT can be adapted for biomedical corpora

BioBERT, a domain-specific language representation model pre-trained on large-scale biomedical corpora

While BERT obtains performance comparable to that of previous state-of-the-art models, BioBERT significantly outperforms them on the following three representative biomedical text mining tasks: 

1. biomedical named entity recognition 
2. biomedical relation extraction 
3. biomedical question answering

recent word representation models trained and tested mainly on datasets containing general domain texts

the word distributions of general and biomedical corpora are quite different, which can often be a problem for biomedical text mining models.

While ELMo and BERT have proven the effectiveness of con- textualized word representations, they cannot obtain high perform- ance on biomedical corpora because they are pre-trained on only general domain corpora

BERT achieves very strong results on various NLP tasks while using almost the same structure across the tasks

adapting BERT for the biomedical domain could potentially benefit numerous biomedical NLP researches

**BioBERT**, which is a **pre-trained language representation model for the biomedical domain**

we **initialize BioBERT with weights from BERT**, which was pre- trained on general domain corpora

Then, BioBERT is **pre-trained on biomedical domain corpora** (PubMed abstracts and PMC full-text articles).

BioBERT is the first domain-specific BERT based model pre- trained on biomedical corpora

pre-training BERT on biomedical corpora largely improves its performance.

most previous biomedical text mining models that are mainly focused on a single task such as NER or QA, our model BioBERT achieves state-of-the-art performance on various biomedical text mining tasks, while requiring only minimal architectural modifications

BioBERT basically has the same structure as BERT

ELMo (Peters et al., 2018) uses a bidirectional language model, while CoVe (McCann et al., 2017) uses machine translation to embed context information into word representations

BERT (Devlin et al., 2019) is a contextualized word representa-
tion model that is based on a masked language model and pre- trained using bidirectional transformers (Vaswani

previous language models were limited to a combination of two unidirectional language models (i.e. left-to-right and right-to- left). BERT

BERT uses a **masked language model** that predicts randomly masked words in a sequence, and hence can be used for learning bi- directional representations.

obtains state-of-the-art perform- ance on most NLP tasks, while requiring minimal task-specific architectural modification. According

incorporating information from bidirectional representations, rather than unidirectional representations, is crucial for representing words in natural language.

*Pre-training BioBERT*

NLP models designed for general purpose language understanding often obtains poor performance in biomed- ical text mining tasks.

we pre-train BioBERT on PubMed abstracts (PubMed) and PubMed Central full-text articles (PMC).

For computational efficiency, whenever the Wiki þ Books corpora were used for pre-training, we **initialized BioBERT with the pre-trained BERT model** provided

We define BioBERT as **a language representation model whose pre-training corpora includes biomedical corpora** (e.g. BioBERT (+ PubMed))

BioBERT uses WordPiece tokenization 

**With WordPiece tokenization, any new words can be represented by fre- quent subwords (e.g.**

using cased vocabulary (not lower- casing) results in slightly better performances in downstream tasks.

we could have constructed new WordPiece vocabulary based on biomedical corpora, we used the original vocabulary of BERTBASE, for these reasons:

1. compatibility of BioBERT with BERT, which allows BERT pre-trained on general domain cor- pora to be re-used, and makes it easier to interchangeably use exist- ing models based on BERT and BioBERT
2. any new words may still be represented and fine-tuned for the biomedical domain using the original WordPiece vocabulary of BERT

*Fine-tuning BioBERT*

With minimal architectural modification, BioBERT can be applied to various downstream text mining tasks. e.g.

* Named entity recognition = recognizing numerous do- main-specific proper nouns in a biomedical corpus. While
  * BERT uses a single output layer based on the representa- tions from its last layer to compute only **token level BIO2 probabilities**
  * while previous works in biomedical NER often used word embeddings trained on PubMed or PMC corpora (Habibi et al., 2017; Yoon et al., 2019), **BioBERT directly learns WordPiece embeddings** during pre-training and fine-tuning



* Relation extraction = classifying relations of named
  entities in a biomedical corpus

  * sentence classifier of the original version of BERT, which uses a [CLS] token for the clas- sification of relations
  * sentence classification is performed using a single output layer based on a [CLS] token representation from BERT
  * anonymize target named entities in a sentence using pre-defined tags such as @GENE\$ or @DISEASE\$.

* Question answering =  task of answering questions posed in
  natural language given related passages

  * . To fine-tune BioBERT for QA, we used the same BERT architecture used for SQuAD
  * used the BioASQ factoid datasets be- cause their format is similar to that of SQuAD. 
  * Token level proba- bilities for the start/end location of answer phrases computed using a single output layer. 

  BioBERT, which is **a pre-trained lan- guage representation model for biomedical text mining.**



### ExplanationLP: Abductive Reasoning for Explainable Science Question Answering - Thayaparan et al. 2020

a novel approach for answering and explaining multiple-choice science ques- tions by reasoning on grounding and abstract inference chains. This

frames ques- tion answering as an **abductive reasoning prob- lem**, constructing **plausible explanations** for each choice and then selecting the **candidate with the best explanation** as the final answer.

**ExplanationLP**, elicits explana- tions by constructing **a weighted graph of rel- evant facts** for each candidate answer and ex- tracting the facts that satisfy certain **structural and semantic constraints**

To extract the ex- planations, we employ a **linear programming formalism designed to select the optimal sub- graph**. The

The **graphs’ weighting function is composed of a set of parameters,** which we fine-tune to optimize answer selection perfor- mance.

state-of-the-art (SOTA) approaches for Science QA are dominated by transformer-based models

these approaches are black-box by nature, lacking of the capability of providing expla- nations for their predictions

**Explainable Science QA** (XSQA) is often framed
as an abductive reasoning problem

**Abductive reasoning** represents a distinct inference process, known as **inference to the best explana- tion** (Lipton, 2004), which **starts from a set of complete or incomplete observations to find the hypothesis, from a set of plausible alternatives, that best explains the observations**. S

XSQA solvers typically treat **ques- tion answering as a multi-hop graph traversal prob- lem**.

the solver attempts to **compose multi- ple facts that connect the question to a candidate answer**. These

These multi-hop approaches have shown diminishing returns with an increasing number of hops ; due to **se- mantic drift** – i.e., as the number of aggregated facts increases, so does the probability of drifting out of context.

e need for a richer representation with fewer hops and higher importance to abstraction and grounding mecha- nisms

**Grounding Facts** that link generic or abstract con- cepts in a core scientific statement to specific terms occurring in question and candidate answer

The grounding process is followed by the identification of the **abstract facts**

Even though a complete explanation for this ques- tion would require the composition of five facts, to successfully derive the correct answer it is pos- sible to reduce the global reasoning in two hops, modeling it with grounding and abstract facts

this work
presents **a novel approach that explicitly models ab- stract and grounding mechanisms by grouping ex- planatory facts into grounding and abstract.** 

These facts are then used to **perform abductive reasoning via linear programming combined with Bayesian optimization.** We

grounding-abstract chains fa- cilitates semantic control for explainable abductive reasoning

Bayesian optimization with linear programming can be employed with few learning parameters and achieve better performance and robustness when compared to transformer-based and graph-based multihop reasoning approaches

ExplanationLP answers and explains multiple-choice science ques- tions via **abductive reasoning**. Specifically, the task of answering multiple-choice science questions is reformulated as the problem of **finding the candi- date answer that is supported by the best expla- nation.**

For each candidate answer ci ∈ C, Ex- planationLP attempts to **construct a plausible ex- planation, quantifying its quality through scoring functions rooted in the semantic relevance of facts while preserving structural constraints imposed via grounding-abstract chains**. The

1. (1) Relevant facts retrieval

* Given a question (Q) and candidate answers, for each candidate answer ci 
  * query the knowl- edge bases using a fact retrieval approach to retrieve the top k relevant facts Fci
  * achieve the retrieval by concatenating question and candidate answer (Q||ci) to retrieve the top l rele- vant grounding facts from a knowledge base containing grounding fact
  * and top m relevant abstract facts Fci
    A = {fci 1 , fci 2 , fci 3 , ..., fci m} from a knowl-
    edge base containing abstract fact

2. Candidate graph construction: 

* For each candidate answer ci we build a weighted undirected graph Gci

3. Subgraph extraction with linear program- ming optimization

* For each graph apply
  linear programming optimization to obtain the optimal subgraph
* The linear programming con-
  straints are designed to emulate the abductive rea- soning over grounding-abstract inference chains
* During the training phase, we update the node and edge weight functions ωv and ωe by tuning the parameters (θ) to optimize for answer selection using Bayesian Optimization

4. answer selection

* apply steps 1-3 for each candidate answer 
* select the one with highest value for final answer

*Constructing Candidate Answer Graph*

to answer a question we need to retrieve
**semantically relevant but still diverse explanations** for each candidate answer.

our approach imposes **several structural and semantic constraints** to tackle this challenge.

**Grounding-Abstract chains** play a cru- cial role in the solution.

scores used:

* relevance
* overlap

construction of the explanatory graphs on the following design principles:
• Explanations for science questions can be con- structed by **multi-hop grounding-abstract chains.**
• Encouraging facts with **higher relevance score limits semantic drift**, as it promotes high concep- tual alignment with the question and answer.
• **Minimizing overlaps in the grounding facts** re- duces semantic drift, as it promotes a higher cov- erage of the concepts in the question.
• **Maximizing overlaps between abstract facts** re- duces semantic drift, as it forces the abstract facts to refer to similar topics, avoiding the construc- tion of contradictory or spurious explanations.
• Encouraging a **high degree of overlaps between hops** prevent the chains to drift away from the original context

*Subgraph Selection with Linear Programming (LP)*

we treat as a **rooted maximum-weight connected subgraph problem** with a maximum number of K vertices

formalism derived from generalized maximum-weight connected subgraph problem 

R-MWCSK has two parts: 

1. objective function to be maximized 
2. constraints to build a connected subgraph. The

The LP solver will seek to extract the opti-
mal subgraph with the highest possible sum of node and edge weights.

In order to emulate the grounding-abstract infer-
ence chains and obtain a valid subgraph, we impose the set constraints stipulated in Table 1 for the LP solver

* Chaining constraint: 
  *  subgraph should always contain the question node.
  *  if a vertex is to be part of the subgraph, then at least one of its neighbors with a lexical overlap should also be part of the subgraph.
  * restrict the LP system to construct explanations that originate from the question and perform multi-hop aggregation based on the existence of lexical overlap. 
  * if two vertices are in the subgraph then the edges connecting the vertices should be also in the subgraph -> force the LP system to avoid grounding nodes with high overlap regardless of their relevance.
* Abstract fact limit constraint: 
  *  limits the total number of abstract facts to K ->  dictate the need for grounding facts based on the number of terms present in the question and in the abstract facts.
* Grounding neighbor constraint: 
  * if a grounding fact is selected, then at least two of its neighbors should be either both abstract facts or a question and an abstract fact -> ensures that grounding facts play the linking role connecting question-abstract facts.
    

the graph built from the correct answer will have the highest node and edge weights.

Unlike neural ap- proaches, **our approach’s gradient function is in- tractable**. In

to learn the optimal values for the parameters, we surrogate the accuracy func- tion with a multi-variate Gaussian distribution N9(µ, σ2) and maximize the accuracy by per- forming **Bayesian optimization**

### Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks - Reimers and Gurevych 2020

BERT (Devlin et al., 2018) and RoBERTa (Liu et al., 2019) has set a new state-of-the-art performance on **sentence-pair regression tasks** like semantic textual similarity (STS).

**requires that both sentences are fed into the network**, which causes a massive com- putational overhead: Finding

The construction of BERT makes it **unsuitable for semantic sim- ilarity search as well as for unsupervised tasks** like clustering

Sentence-BERT (SBERT), a modification of the pretrained BERT network that use **siamese and triplet net- work structures to derive semantically mean- ingful sentence embeddings that can be com- pared using cosine-similarity**.

Sentence-BERT (SBERT), a modification of the BERT network us- ing siamese and triplet networks that is able to derive semantically meaningful sentence embed- dings

s enables BERT to be used for certain new tasks, which up-to-now were not applicable for BERT, e.g.

* large-scale seman-tic similarity comparison, 
* clustering,
*  informa- tion retrieval via semantic search.

BERT uses a **cross-encoder**: **2 sentences are passed to the transformer** network and the target value is predicted

unsuitable for various pair regression tasks due to too many possible combinations

A common method to address clustering and se-
mantic search is to **map each sentence to a vec- tor space such that semantically similar sentences** 

Researchers have started to **input indi- vidual sentences into BERT and to derive fixed- size sentence embeddings**

* most commonly used approach is to average the BERT output layer (known as BERT embeddings) or by using the out- put of the first token (the [CLS] token).
* yields rather bad sentence embeddings, often worse than averaging GloVe embeddings

SBERT: siamese network architecture enables that fixed-sized vectors for input sentences can be de- rived.

* similarity measure like cosine- similarity or Manhatten / Euclidean distance, se- mantically similar sentences can be found. These
* can be performed extremely efficient on modern hardware, allowing SBERT to be used for semantic similarity search as well as for clustering. 

SBERT can be adapted to a specific task

**The input for BERT for sentence-pair regression consists of the two sentences, separated by a special [SEP] token**

Multi-head attention over 12 (base-model) or 24 layers (large-model) is applied and the out- put is passed to a simple regression function to de- rive the final label.

RoBERTa (Liu et al., 2019) showed, that the performance ofBERT can further improved by small adaptations to the pre-training process

A large disadvantage of the BERT network structure is that **no independent sentence embed- dings are computed**, which makes it difficult to de- rive sentence embeddings from BERT

To bypass this limitations, researchers passed single sen- tences through BERT and then derive a fixed sized vector

*  by either averaging the outputs (similar to average word embeddings) or 
* by using the output of the special CLS token

**Sentence embeddings** are a well studied area
with dozens of proposed method, e.g.

* train an encoder-decoder ar- chitecture to predict the surrounding sentences (Skip-Thought )

*   use labeled data to train a siamese BiLSTM network with max-pooling over the output (InferSent)
* train a transformer network and augments unsupervised learning with training on SNLI (Universal Sentence Encoder)
* train on conversations from Reddit using siamese DAN and siamese transformer net- works

the task on which sentence embeddings are trained significantly impacts their quality

SNLI datasets are suitable for training sen- tence embeddings. 

Humeau et al. (2019) and present 

a method (poly-encoders) to compute a score between m context vectors and pre-computed candidate embeddings using attention

* addresses the run-time
  overhead of the cross-encoder from BERT 
* works for finding the highest scoring sentence in a larger collection
* poly- encoders drawback that the score function is not symmetric and the computational overhead is too large for use-cases like clustering

Previous neural sentence embedding methods
started the training from a random initialization.

we use the **pre-trained BERT and RoBERTa** network and only fine-tune it to yield useful sentence embeddings.

This reduces significantly the needed training time

**SBERT adds a pooling operation to the output of BERT / RoBERTa to derive a fixed sized sen- tence embedding**

3 pool- ing strategies: 

1. Using the output of the CLS-token, 
2. computing the mean of all output vectors (MEAN- strategy), (default)
3.  computing a max-over-time of the output vectors (MAX-strategy). 

 **fine-tune** BERT / RoBERTa: cre-
ate **siamese and triplet networks to update the weights** such that the produced **sentence embeddings are semantically meaningful and can be compared with cosine-similarity**

network structure depends on training data

* Classification Objective Function
  * con-
    catenate the sentence embeddings u and v with the element-wise difference |u−v| and multiply it with the trainable weight
  * optimize cross-entropy loss.
* Regression Objective Function. The
  * cosine- similarity between the two sentence embeddings u and v is computed 
  *  mean- squared-error loss as the objective function
* Triplet Objective Function
  * Given an anchor sentence a, a positive sentence p, and a negative sentence n, triplet loss tunes the network such that the distance between a and p is smaller than the distance between a and n. 

BERT out-of-the-box maps sen- tences to a vector space that is rather unsuit- able to be used with common similarity measures like cosine-similarity; below the performance of average GloVe embeddings.

SBERT fine-tunes BERT in a siamese / triplet network architec- ture. We

could achieve a sig- nificant improvement over state-of-the-art sen- tence embeddings methods

. Replacing BERT with RoBERTa did not yield a significant improvement



### Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge - Clark et al. 2018

a new question set, text corpus, and baselines assembled to encourage AI research in advanced question answering. Together,

The ARC Dataset consists of a collection of 7787 nat-
ural science questions, namely questions authored for use on standardized tests.

these questions are difficult to answer via simple re- trieval or word correlation.

The ARC dataset consists of **7787 science questions, all non-diagram, multiple choice** (typically 4-way multiple choice). They

### WorldTree V2: A Corpus of Science-Domain Structured Explanations and Inference Patterns supporting Multi-Hop Inference  - Xie et al. 2020

Explainable question answering for complex questions often requires combining large numbers of facts to answer a question while providing a human-readable explanation for the answer, a process known as **multi-hop inference**. 

Standardized science questions require combining an average of 6 facts, and as many as 16 facts, in order to answer and explain, but most existing datasets for multi-hop reasoning focus on combining only two facts, significantly limiting the ability of multi-hop inference algorithms to learn to generate large inferences. 

WorldTree project, a corpus of **5,114 standardized science exam questions paired with large detailed multi-fact explanations** that combine core scientific knowledge and world knowledge.

Each explanation is represented as **a lexically-connected “explanation graph” that combines an average of 6 facts drawn from a semi-structured knowledge base of 9,216 facts across 66 tables.**

author a set of 344 high-level science domain inference patterns similar to semantic frames supporting multi-hop inference

**Explainable question answering** is the task of providing both answers to natural language questions, as well as detailed human-readable explanations justifying why those answers are correct

Question answering is typically approached using either retrieval or inference methods, 

*  **retrieval methods** search for a single contiguous passage of text from a corpus or single fact in a knowledge base that provides an answer to a question. 
* For complex questions, a single passage often provides only part of the knowledge required to arrive at a correct answer, and an **inference model** must combine multiple facts from a corpus or knowledge base to infer the correct answer

Combining facts to perform inference is an inherently noisy process that often drifts off-context to unrelated facts, a phenomenon referred to as **semantic drift** 

* most multi- hop inference models are generally unable to demonstrate combining more than 2 or 3 facts to perform an inference

* reasoning required to answer elementary science exams averages combining 6 separate facts, and as many as 16 facts, when

we present a large corpus of extremely
detailed multi-fact explanations to serve both as training data for multi-hop inference, as well as an instrument to evaluate and expand the information aggregation capacity of multi-hop inference models

### CancerBERT : a BERT model for Extracting Breast Cancer Phenotypes from Electronic Health Records - Zhou et al. 2021

The word embeddings obtained by Word2Vec and Glove models are **static word embeddings**, each token in the model is represented by a unique word vector with fixed dimension and values

s, the static word embeddings are **incapable of reflecting all possible meaning of a word according to its contextual information.**

This improved type of word embedding is **contextual word embedding**. Currently, the most powerful contextual word embedding model is the BERT 12.

BERT applied a **bidirectional transformer architecture as encoder** **to encode the input sentences**.

The transformer architecture uses **multiheaded self-attention to avoid the locality bias and to better capture the long-distance context information**

compared to the traditional LSTM architecture, the self-attention could **compute in a parallel way**, which makes it compute efficiently.

T, we applied the f**ine-tuning** method that integrates the pre-trained BERT model into the downstream NER model. During the fine-tuning process, the **parameters of pre-trained BERT model and downstream NER model will be updated simultaneously** to finish the NER task.

we trained a cancer domain specific BERT model (CancerBERT) that is expected to better capture the semantics of the cancer specific clinical notes and pathology reports, and may improve the performances of the NER task for extracting breast cancer related phenotypes

We also revised the **vocabulary** for the original BERT model. Since the original vocabulary was developed based on the corpus in general domain (e.g., Wikipedia), many special words and abbreviations exist in the clinical narratives cannot be covered. The

The **out of vocabulary (OOV) issue** may influence the performances of the language model

**WordPiece tokenizer** was applied to deal with the OOV issue. It **tokenizes an unknown word into multiple sub-words that exist in the vocabulary**. For instance, the word “HER2”, a breast cancer related cell receptor gene, is not in the original BERT vocabulary, and it will be tokenized into “HER” and “2” by the WordPiece tokenizer, and will use **the average of their word embeddings** to represent “HER2”

we added 397 cancer related new words into the original BERT vocabulary

### Attention is all you need - Vaswani et al. 2017

We propose a new simple network architecture, the **Transformer**, **based solely on attention mechanisms**, dispensing with recurrence and convolutions entirely. Experiments

Recurrent models typically factor computation along the symbol positions of the input and output sequences. Aligning the positions to steps in computation time, they generate a sequence of hidden states ht, as a function of the previous hidden state ht−1 and the input for position t. This inherently sequential nature precludes parallelization within

**Attention mechanisms** allowing modeling of dependencies without regard to their distance in the input or output sequences

the **Transformer**, a model architecture eschewing recurrence and instead **relying entirely on an attention mechanism to draw global dependencies** between input and output

allows for significantly **more parallelization**

convolutional neural networks as basic building block, computing hidden representations in parallel for all input and output positions; the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions,; more difficult to learn dependencies between distant positions

In the Transformer this is reduced to a **constant number of operations**, albeit at the cost of **reduced effective resolution due to averaging** attention-weighted positions, an effect we **counteract with Multi-Head Attention**

**Self-attention**, sometimes called **intra-attention** is an attention mechanism **relating different positions of a single sequence in order to compute a representation of the sequence**

**End-to-end memory networks** are **based on a recurrent attention mechanism** instead of sequence- aligned recurrence and

Transformer is the first transduction model **relying entirely on self-attention** to compute representations of its input and output without using sequence- aligned RNNs or convolution.

Most competitive neural sequence transduction models have an **encoder-decoder structure**

the **encoder** maps an input sequence of symbol representations to a sequence of continuous representations **z**

At each step the model is **auto-regressive** [10], consuming the previously generated symbols as additional input when generating the next

The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown

*Encoder:* 

* stack of N = 6 identical layers
* Each layer has 2 sub-layers
  1. multi-head self-attention mechanism
  2. simple, position- wise fully connected feed-forward network.
*  residual connection around each of the two sub-layers,
*  followed by layer normalization 
*  output of each sub-layer is LayerNorm(x + Sublayer(x)),
  *  Sublayer(x) is the function implemented by the sub-layer itself. 
* To facilitate these residual connections, all sub-layers as well as the embedding layers, produce outputs of dimension dmodel = 512.

*Decoder*

* also composed of a stack ofN = 6 identical layers
* . In addition to 2 two sub-layers in each encoder layer, inserts a 3d sub-layer: performs **multi-head attention over the output of the encoder** stack
* also residual connections around each of the sub-layers, followed by layer normalization
* **modify the self-attention** sub-layer in the decoder stack to **prevent positions from attending to subsequent positions**
  *  This **masking, combined with fact that the output embeddings are offset by one position**, ensures that the **predictions for position i can depend only on the known outputs** at positions less than i

*Attention*

An **attention function** can be described as **mapping a query and a set of key-value pairs to an output,** where the query, keys, values, and output are all vectors.

* The output is computed as a **weighted sum of the values**, 
* the weight assigned to each value is computed by a **compatibility function of the query with the corresponding key**

### Differentiating through a cone program - Agrawal et al. 2019

A **cone program** is an optimization problem in which the objective is to minimize a linear function over the intersection of a subspace and a convex cone. every convex optimization problem can be expressed as a cone program

An optimization problem can be viewed as a (possibly multi-valued) function mapping the problem data, i.e., the numerical data defining the problem, to the (primal and dual) solution. This
