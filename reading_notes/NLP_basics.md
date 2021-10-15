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