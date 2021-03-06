### Context ‑ aware multi ‑ token concept recognition of biological entities - Kim and Lee 2021

Concept recognition is a term that corresponds to the two sequen- tial steps of 

1. named entity recognition (NER)
2. named entity normalization (NEN)

a concept recognition method of multi-token bio- logical entities using neural models combined with literature contexts

The key aspect of our method is utilizing the contextual information from the biological knowledge- bases for concept normalization, which is followed by named entity recognition procedure.

In NER, or text span detection stage, the location and the classified type of the entity mention in the given text is identified

NEN is also referred as various names, including entity disambiguation, entity linking or entity mapping. The

The common goal of this procedure is to identify a biological concept in the knowledgebase corresponding to the text span found in the NER stage and link it to an ontology identifier.

there are vast amount of knowledge accumulated in the biological knowledge-bases, this knowledge cannot be utilized by performing NER alone

the text span should be ’linked’ or ’mapped’ into specific entity of the biological knowledgebase to derive meaningful biological information from text

the connection between the text spans and specific entities within the database should be identified as a form of certain ID (“GO0008286”). This

The biggest factor that hinders the recognition of multi-token concepts is name varia- tion.

Gene Ontology [1] contains six synonyms for the biological process concept “positive regulation of biological process”, which is clearly insufficient to reflect the extensive potential variation of the term. Due to this factor, conventional dictionary-based concept recognition methods showed insufficient perfor- mances on concepts with relatively longer, multi-token terms

, we have extracted the contextual information from the biological knowledge-bases to infer the improved normalization for the biological concepts

The key feature of BERT is application of bidirectional attention-based transformer.

The BERT model is trained on two tasks: first is the masked language task, which use the output of the masked word’s position to predict the masked word. The other is the next sentence prediction task, which predict likelihood of a sentence belongs after another sentence. The

there isn’t an established BERT model for entity normaliza- tion

This is due to the nature of the BERT language model, where the vector value of the word piece token is determined by the context of the surrounding, making it difficult to derive the vector of every entity mention in the text in advance

semantically similar biological concepts cannot be assumed to have similar identifiers, and the biological concepts with similar identifiers cannot be assumed to have similar meanings

‘end-to-end’ normalization approach; , the model does not train NER and NEN as separate sequential steps. Instead, it is trained as a single joint task

the method basically considers the normalization process as multi-class classification task. That is, the model cannot label the ‘unseen concepts that were not appeared in the training set. This limitation reduces the advantage of the model to be used for general tasks for broader dataset

, there
have not been many approaches that utilized **contextual information of the ontology**. In this study, we have developed a concept recognition methods utilizing these vari- ous contextual information that can be accessed from the ontology structur

procedure:

1. NER module based on bioBERT language model to identify the text spans of the mentions according to known biological concepts from the input text. 
2. derived the vector representations of the obtained concept mentions with given text mentions, and concept names and definitions texts of the candidate biolog- ical concepts. 
3. Using the cosine similarities among vector represented concept men- tions, concept names and concept definitions, the normalized concept was predicted

*NER*

utilized the NER module provided by bioBERT model for NER step.

Input text is tokenized into Wordpiece tokens [12], then fed into pre-trained bioBERT model

The model is fine-tuned to predict the label corresponding to each tokens. In

*concept normalization*

Concept normalization, we have derived vector representations of each text spans of the mentions that were obtained in the NER stage, concept titles and concept defi- nitions of knowledgebase

*Structure of ontology concepts*

the open biomedical ontologies (OBO) format [13].

we have selected is_a and part_of relations as those relations occupy majority of the ontology and conveys the hierarchical relation we intend to extract.

**Sentence-BERT** [14], a BERT-based model fine-tuned for semantically meaningful sentence embedding. While

**BERT base models : the vectors of each word tokens are not fixed on a particular value. As a result, the base BERT model is not capable of predicting vec- tors of untrained sentence queries**

**The main advantage of utilizing the Sentence-BERT**
**model is that vector representations based on the BERT model can also be derived for**
**input query text given without additional fine-tuning.**

The Sentence-BERT model is built on the Siamese network and is trained to predict the similarity score between two input sentence vectors. That

, the model is trained to output high similarity scores for similar sentences. As

the Siamese network model is able to predict a representation vector corresponding to the input sentences through metric learning, which can be used to calculate similarity between any arbitrary sentence pairs

vector representations for the extracted mention, concept names, concept definitions, and hierarchical contexts were derived.

the normalization model cosine similarities of each mention-context pairs

Then the cosine similarities between mention-name, mention-definition and mention-hierarchy pairs were concat- enated to form a feature vector

we constructed a regres- sion model using feature vectors as input data and mention-concept pair extracted from CRAFT corpus as golden standard set

### Extracting comprehensive clinical information for breast cancer using deep learning methods - Zhang et al. 2019

Our system consists of two components: a named entity re- cognition (NER) component and a relation recognition component. For each component, we implemented deep learning-based approaches by fine-tuning BERT

. A clinical language model is first pretrained using BERT on a large-scale unlabeled corpus of Chinese clinical text. For

For NER, the context embeddings that were pretrained using BERT were used as the input features of the Bi-LSTM-CRF model and were fine-tuned using the annotated breast cancer notes.

we proposed an approach to fine-tune BERT for relation extraction

It performs better than previous methods because it applies the first attention-based bi- directional neural network architecture to jointly represent the se- quence information in both directions for language models.

Our system consists of three components: preprocessing, NER and
relation extraction

the BI-LSTM-CRF is
widely used and achieves good performances in different NER tasks

The BI-LSTM-CRF uses a BI-LSTM to score all possible labels
for each token in a sequence, and it predicts a token’s label using its
neighbor’s information in a CRF layer

The output vectors of the pretrained language model using BERT are used as the features of the input layer of the BI-LSTM-CRF

. Both the BI-LSTM-CRF network and the BERT network are tuned during the training process

Given the annotated entities in sentences, the relation extraction
task can be transformed into a classification problem. A classifier can be built to determine the categories of all possible candidate relation pairs (e1, e2), where entities e1 and e2 are from the same sentence. We gen- erated candidate pairs by pairing each concept and another entity with a semantic type that matches an attribute of the concept.

BERT adds a classification token [CLS] at the beginning of a sentence input, and the output vector was used for classification. As is typical with BERT, we used a [CLS] vector as the input to the linear layer for classification. Then, a softmax layer was added to output labels for the sentence.

### Explainable Transformer-Based Neural Network for the Prediction of Survival Outcomes in Non-Small Cell Lung Cancer ( NSCLC ) - Kipkogei et al. 2021

“Clinical Transformer” - a recasting of the widely used transformer architecture as a method for precision medicine to model relations between molecular and clinical measurements, and the survival of cancer patients.

the emergence of immunotherapy offers a new hope for cancer patients with dramatic and durable responses having been reported, only a subset of patients demonstrate benefit. Such treatments do not directly target the tumor but recruit the patient’s immune system to fight the disease. Therefore, the response to therapy is more complicated to understand as it is affected by the patient’s physical condition, immune system fitness and the tumor. As

**As in text, where the semantics of a word is dependent on the context of the sentence it belongs to, in immuno- therapy a biomarker may have limited meaning if measured independent of other clinical or molecular features.**

we hypothesize that the transformer-inspired model may potentially enable effective modelling of the semantics of different biomarkers with respect to patients’ survival time. Herein,

offer an attractive alternative to the survival models utilized in current practices

1. an embedding strategy applied to molecular and clinical data obtained from the patients
2. a customized objective function to predict patient survival
3. applicability of our proposed method to bioinformatics and precision medicine

the clinical transformer outperforms other linear and non-linear methods used in current practice for survival prediction. We

when initializing the weights of a domain-specific transformer by the weights of a cross-domain transformer, we further improve the predictions. Lastly, we show how the attention mechanism successfully captures some of the known biology behind these therapies.

we extend to the domain of precision medicine and ask whether it is possible to predict treatment outcome and response from individualized records of patients undergoing immuno-oncology interventions

the single patient level, these clinical and molecular features conceal complex interrelationships that are intertwined with the patient outcome. The

small sample size which poses a great challenge to building reliable and interpretable predic- tion models

the complexity of the underlying biology and the heterogeneity of the patient population impair the ability to define biomarkers that can be used for patient selection in subsequent trials.

the ability to utilize **transfer learning**, an attribute that comes as part of the transformer architecture, is of great importance

transferable representations enable us to **extend automated survival analysis** to conditions where limited patient data is available in a manner that was not possible in traditional approaches.

Survival analysis is an area that has been dominated by statistical approaches such as multi-variate modelling using Cox proportional hazards (CoxPH) for predicting patients’ response to therapy based on individual’s genomic and clinical features

Each patient entry included genomic information of 468 genes (those are, mutation calls for each of the 468 genes) and 8 clinical features (e.g. age and sample type

A cross-domain clinical transformer (Figure 1B) was pre-trained on 10 cancer types (N=1266)

specific clinical transformer (Figure 1C) was trained with 80% of the NSCLC data excluding; This transformer was initialized with the weights from the pan cancer transformer.

Transformer attentions capture interactions between input features in the context of patients’ survival (Figure 1E). These attention weights can be used for different bioinformatics applications, including network analysis in systems biology, gene set enrichment to identify dominant biological pathways with clinical relevance, clustering and other pattern recognition methods for patient stratificatio

since we are not dealing with sequential data, we excluded the positional encoding vectors from the transformer architecture

each component of the feature space F is decomposed into a linear combination of the learned weights that is then fed into the transformer. The rationale behind this decomposition is to embed numerical data into a fixed vector size

The **scaled dot-product attention** in a transformer, enables the model to selectively focus on relevant features from the input space by identifying similarities among the input features while associating those similarities with the model outcome

To optimize model parameters towards patient survival outcomes, instead of a binary response or text translation, we utilized the concordance metric in survival analysis workflow as a measure of model discrimination

Harrell’s concordance index C is defined as the proportion of observations that the model can order correctly in terms of survival times ( Steck et al. [2008]). This can be interpreted as a generalization of the area under the ROC curve (AUC) that considers censored data. It represents the global encapsulation of the model discrimination power and its ability to provide a reliable ranking of the survival times based on the individual risk scores. In

The predicted scores from the transformer are directly proportional to patients’ survival time and therefore the scores can be analyzed using standard survival analysis approaches such as the concor- dance index (note

potential advantage of transfer learning to improve the ability to use the clinical transformers in clinical settings where a relatively small data set is available for analysis

**attention weights in the clinical transformer could be used to measure the strength of association of the variables with other variables and their effect on the prediction of the clinical outcome.**

we propose the **Variable Interaction Score** (VIS) that considers the interactions among the input features and the association with survival.

the VIS can be calculated at the single patient level, this metric can capture in addition to prevalent population of patients, a rare event – a small group of patients with a strong prognostic or predictive molecular or clinical characteristics.

we examined whether the CoxPH and the clinical transformer models accorded similar levels of importance to the input features. Although the two models are different in terms of their underlying modelling assumptions and complexity level, we expected to see some level of agreement in variable rankings, especially in well-known biomarkers in Immunotherapy. A summary of the comparison results is shown in Figure S4. We found a reasonable concordance in the importance levels of the variables between the two models (Pearson

to test the possible connection of the attention score with the underlying tumor biology considering the survival time of the patients.

A primary interest in clinical trials is to identify biomarkers for a given treatment that can predict fast-progressor and supper-responders

we split the patient population into four ‘survival’ groups. Since these four groups respond differently to the treatment, biological functions that are known to be strongly associated with cancer progression may contribute to these sub-populations in different magnitude.

we grouped the genomic features (that is, the genes) in our data based on their known function to two different key functionalities that are strongly related to cancer progression; onco-suppressors genes ( Repana et al. [2019]) and growth pathways genes

we grouped three key clinical features together to assess their differential interactions across the four patients’ sub-populations.

these four feature groups demonstrate monotonic decrease (Figure 6A and Figures S8A,S8B) or increase (Figure 6B) as a function of patient survival.

### ASBERT: Siamese and Triplet network embedding for open question answering - Shonibare 2021

A common approach to address the AS problem is to generate an embedding for each candidate sentence and query. Then, select the sentence whose vector representation is closest to the query’s. A

A key drawback is the low quality of the embeddings, hitherto, based on its performance on AS benchmark datasets. In

ASBERT, a framework built on the BERT architecture that employs Siamese and Triplet neural networks to learn an encoding function that maps a text to a fixed-size vector in an embedded space.

The notion of distance between two points in this space connotes similarity in meaning between two texts

question answering can be differentiated in two major ways: Open-domain question answering (QA) and Closed-domain question answering (QA). While

open-domain QA system typically comprises the following pipeline stages: A user question is parsed to determine its type and/or extract keywords; find suitable documents from a very large corpora; for each selected document, identify candidate answer sentences; lastly, determine a subset of the list that most likely contains the answer to the given question. In

In this work, our primary focus is on the task of selecting and ranking plausible answers to a given question from a set of candidate sentences, which is often referred to as answer election. 

A key difficulty associated with this effort is that the most suitable answer sentence might contain a lot of unrelated information and shares very few lexical features with the question sentence

Distributed word encodings have been demonstrated to be effective in multiple NLP tasks. However, there are some classes of problems where these word vector representation are inadequate.

, sentence encodings are desired for a better language comprehension. State-of-the-art

State-of-the-art models like BERT [8], RoBERTa [28] and XLNet [29] have shown great performance in several benchmark tasks including question answering, however, it can be **computationally expensive when used for text regression problems since it involves a huge number of sentence comparisons.**

**the embeddings generated either by averaging the output vectors or using only the vector that corresponds to the first token, [CLS], for the case of a single sentence input, has been shown to be of low quality** [16].

ASBERT, a deep learning framework based on BERT that **utilizes both siamese and triplet networks** to learn useful sentence encoding

this work appears to be the first to evaluate the embeddings obtained from a siamese/triplet network built on a BERT architecture for AS problems

The **Siamese and the Triplet networks**. The underlying goal of both networks is to learn a semantic embedding space into which an input text could be mapped.

given a question sentence and a candidate answer sentence, a distance function can be applied to their vector representation to determine their closeness. The

The term, **closeness**, in this work refers to how well the candidate sentence answers the question. The

For both architectures, LM is a pretrained language model which is based on a transformer architecture. The **Transformer** comprises many attention blocks and within a block, an input vector is transformed by a combination of a self-attention layer and a feed-forward neural network in that order, where the self-attention layer incorporates the influence of neighboring words to the encoding of a particular word.

The forward propagation begins with the computation of three embeddings: the token, the position and the segment embeddings

* The token embeddings are obtained by splitting the input sentence into tokens and then each token is replaced with their corresponding index in the tokenizer vocabulary;
* the position embeddings represents the relative position of each token within a sentence 
* the segment embeddings is used to address a situtaion where the input is composed of two sentences and is obtained by assigning the tokens for each sentence a unique single label e.g. 0 and 1. 
* The final input is then obtained by adding these embeddings. 

This sequence is mapped through several layers up to the last. 

The final output, which is the same size as the input embeddings is passed to a pooling layer (P1). Some

Some examples of pooling operations that can be applied here includes, max, mean, extracting only the sequence that corresponds to the first token ([CLS]) or even applying attention to the sequence.

In this work, the mean pooling operation is adopted. 

The result of the pooling operation in P1 is then passed on to a number of layers downstream before finally applying a loss function. 

Once the loss optimization procedure is complete, the trained base network LM-P1 would have learnt a function that is able to generate discriminative features, given any sentence, such that **the distance between a question and a positive answer sentence is small in the embedding space** and large for the same question and negative answer sentence. The

*Siamese network*

Each training example comprises a question sentence, a candidate answer sentence and a label indicating whether it is a positive match (1) or a negative match (0). The

The two LM modules are essentially copies of the same network. They share the same parameters.

*Triplet network*

a triplet network consist of three replicas of a feedforward network that share identical weights. It

It accepts three input sentence, a positive, an anchor and a negative sentence, generates an encoding at layer P1 and then computes the distance between the anchor sentence and positive sentence, and anchor sentence and negative sentence.

### A Sui Generis QA Approach using RoBERTa for Adverse Drug Event Identification - Jain et al. 2020

a question answering framework that exploits the robustness, masking and dynamic attention capabilities of RoBERTa by a technique of domain adaptation

the multi-head self attention capabilities allowed transformers to capture long range dependencies effi- ciently. This along with contextual dense representations from pre-training using a very large corpus allowed a **better feature extraction capability**

Transformer networks utilize the power of **multi-head self-attention mechanism** to capture context-sensitive embeddings and interactions between tokens. Since

a new QA frame-
work for ADE identification task using a more powerful transformer architecture RoB- ERTa

NER module: to identify the drug names in a given phrase, we leveraged recently developed Med7 [16] NER module which

Classification module: After rec- ognizing the drug entities in entity recognition module,in order to identify the phrases where at least one drug and adverse event pair coexists at stage 2, we trained a Bi-LSTM [17] based binary classifier on ADE sentences and cross-validated it in K-Fold setting.

Q&A module : a question-answering system discovers a span of text in the passage that best describes answer to the question being asked

in their paper that introduces RoBERTa, observed **BERT to be “signifi-**
**cantly undertrained”** and also found out that with a better selection of hyperparamerters and training size, its performance could be considerably improved

RoBERTa base model and fine-tune it on drug-related adverse effects corpus
to identify the adverse event corresponding to a drug. To

. To perform fine-tuning, we utilize pre-trained weights of RoBERTa’s 12 layer transformer network and add a CNN based Q&A head (Fig. 2) on top of it

we process the data in desired input format for RoB- ERTa into 2 segments A and B. 

* Segment A consists an encoded vector of drug treated as a question 
* segment B that consists another encoded vector of context/ sentence where adverse event is mentioned.

we pass this processed data into a 12-layered transformer network of RoBERTa and use its output that represents the 768 dimensional learnt embeddings of encoded input for further processing.

we apply a one dimensional CNN layer with a (1 x 1) convolution filter that
creates a feature map of these embeddings followed by a flatten and a softmax activa- tion layer to predict the probability of start/end tokens of the adverse event present in a
span of the given text.

Since **1D CNN layer renders the network with property of being “translationally invariant”, a pattern of adverse event learnt at one position in a sentence can be easily identified at a different position as the same input transformation is applied on every patch**

### Bert-based siamese network for semantic similarity - Feifei 2020

Aiming at the insufficient feature extraction capabilities of traditional Siamese-based text matching methods, a fast matching method based on large-scale pre- training model BSSM (BERT based Siamese Semantic Model) is proposed

The proposed method uses a pre-training model to encode two texts separately, which interact the representation vectors of the two texts to obtain attention weights and generate new representation vectors, so that the new and old representation vectors can be pooled and aggregated, and finally two representation vectors are concatenated in some strategy and sent to the prediction layer for prediction. Experimental

What needs to be done when a user asks a question is to calculate the similarity between the user question and the configured standard question. Then the standard question most similar to the user's question can be selected, and the corresponding answer to the user will be returned. In

for text matching tasks, we should not only focus on matching at the lexical level, but
also at the semantic level. 

the same semantics can also be expressed by different words, for example, "Apple mobile phone" and "iphone" have the same meaning

Using the word embedding obtained by training the neural network language model to perform text matching calculation, the training method is more concise, as well as the semantic computability of the word embedding representation obtained is further enhanced.

2 types of methods for obtaining word embedding: 

1.  local context window method, such as the CBOW and Skip-Gram methods 
   * based on a local context window, so the global lexical co-occurrence statistics cannot be effectively used.
2. method based on global matrix decomposition, which is named LSA
   * although LSA effectively uses statistical information, it is very poor in terms of vocabulary analogy,

the word embedding itself does not solve the problem of semantic representation of words and sentences, and the semantics cannot be changed with the change of context

this paper proposes a Siamese network semantic model based
on BERT (Bidirectional Encoder Representation from Transformers). By dynamically adjusting word representation through BERT, it can effectively solve the problem of semantic representation of sentences,

the text matching method in the deep learning era can be summarized into three types: 

1. representation model, 
   * The text is converted into a representation vector through the presentation
     layer, so this method focuses more on the construction of the presentation layer.
   * The Siamese structure is used to extract the semantics of the text separately and then match. 
   * The parameters of the two towers are shared, and networks such as MLP, CNN, RNN, Transformer, etc. can be used
   * shared parameters make the model smaller and easier to train. The
   * no clear interaction between the two texts in the mapping process, where a lot of information about each other is lost
   * Representations are used to represent high-level semantic features of text, but high-level representations of features such as word relationships and syntax in the text are more difficult to capture, and it is difficult to determine whether a representation can well represent a piece of text
2. interaction model 
   * The interaction model abandons the idea of post-matching.
   * the first matching between words is performed in the interaction layer, and the interactive results are subsequently modeled.
   * This method does not directly learn the semantic representation vectors of the two sentences, but at the bottom layer, it lets the two sentences interact in advance to establish some basic matching signals, and then find a way to fuse these basic matching signals into a matching score. The
   * This framework captures more interactive characteristics between two sentences, so it significantly improves compared to the first framework.
   * neglects the global information such as syntax, and the global matching information cannot be depicted by local matching information.
3. large-scale pre-training model
   * In the early days of natural language processing, word embedding methods such as Word2Vec were used to encode text for a long time. These word embedding methods can also be regarded as static pre- training techniques. However, this context-free text representation brings very limited improvement to subsequent natural language processing tasks, and cannot solve the problem of ambiguity. 
   * BERT uses Transformer's Encoder, the main innovation of which lies in the pre-train method,
     that is, it uses Masked Language Model (MLM) and Next Sentence Prediction to capture word and sentence-level representations respectively. 

this article proposes a new architecture BSSM (BERT based
Siamese Semantic Model) suitable for text matching. The model uses BERT to encode two texts to obtain word embedding. Let the representation vectors of the two texts interact to obtain attention weights and generate new word embedding. Then the new and old word embedding are pooled and aggregated, and finally the two word embedding are concatenated and sent to the final prediction layer for prediction.

Because of the Siamese architecture, the texts to be matched can be encoded in advance, which can greatly reduce the matching time. And

### EPICURE: Ensemble pretrained models for extracting cancer mutations from literature - Cao et al. 2021

EPICURE, an ensemble pre- trained model equipped with a conditional random field pattern (CRF) layer and a span prediction pattern (Span) layer to extract cancer mutations from text. We

We also adopt a data augmentation strategy to expand our training set from multiple datasets

we suggest methods for extracting mutation-
disease associations found in genetic studies by automatically mining the scientific literature

named entity recognition(NER) methods to extract information relevant for cancer mutation detection tasks

due to the limited availability of existing cancer mutation datasets, BiLSTM-based CRF models [11]–[13] do not perform well in cancer mutation tasks since these models rely on massive volume of training data.

We propose an ensemble pre-trained model that combines
a CRF pattern and a span-prediction pattern to automatically extract the mutations from cancer literature without depending on any hand-crafted linguistic features or rules. We

We first design two model patterns, where the first one predicts mutation entities in a token-by-token manner with a CRF layer [22], and the second identifies mutation entities by predicting the entire possible spans with a pointerNet [32].

Then we introduce the pre-trained model as our model encoder to leverage prior knowledge from massive pre-trained corpora. Furthermore,

the model structure, of which it constitutes a BERT-based [14] encoder layer and two separate model patterns in the upper layer denoting two different methods of identifying mutations. Finally, the model resulting from two CRF and span patterns can be incorporated using majority voting

### Semi-Automating Knowledge Base Construction for Cancer Genetics - Wadhwa et al. 2020

We propose two challenging tasks that are critical for characterizing the findings re-
ported in penetrance studies: (i) Extracting snippets of text that describe ascertainment mechanisms, which in turn inform whether the population studied may introduce bias ow- ing to deviations from the target population; (ii) Extracting reported risk estimates (e.g., odds or hazard ratios) associated with specific germline mutations. The

To train models for these tasks, we induce distant supervision over tokens and snippets in full-text articles using the manually constructed knowledge base

### Structure-aware Sentence Encoder in Bert-Based Siamese Network - Peng et al. 2021

we show that **by incorporating structural information into SBERT,** the resulting model outperforms SBERT and previous general sentence en- coders on unsupervised semantic textual simi- larity (STS) datasets and transfer classification tasks

Relational Graph Convo- lutional Networks (**RGCNs**) to incorporate syntactic dependency graphs

structural supervision is useful, and that RGCNs serve as an effective structure encoder.

BERT can also be used as a general sentence
encoder, either by using the CLS token (the first token of BERT output) or applying pooling over its output

However, this fails to produce sentence embeddings that can be used effectively for similar- ity comparison. 

Furthermore, this method of using BERT for similarity comparison is extremely inef- ficient, requiring sentence pairs to be concatenated and passed to BERT for every possible comparison

Sentence-BERT (**SBERT**) has been proposed to alleviate this by fine-tuning BERT on natural language inference (NLI) datasets using a siamese structure (Reimers

it is possible to im-
prove the SBERT sentence encoder through the use of explicit syntactic or semantic structure.

we propose a model that combines the two by training a BERT-RGCN model in a siamese structure

Under specific structural supervision, the proposed model is able to produce structure- aware, general-purpose sentence embeddings.

it outperforms SBERT and previous sentence encoders on unsupervised similarity comparison and transfer classification tasks. Furthermore,

we train our model in a siamese network to update weights so as to produce similarity-comparable sentence representations

BERT: Each sentence is first fed into the pre- trained BERT-base model to produce both a sen- tence representation, by applying mean-pooling, and an original contextualised sequence-length to- ken representation, which is used to initialise a RGCN.

3 Model
Inspired by Reimers and Gurevych (2019), we train our model in a siamese network to update weights so as to produce similarity-comparable sentence representations. The model we propose consists of two components, as shown in Figure 1.
Figure 1: The proposed model in siamese structure
BERT: Each sentence is first fed into the pre- trained BERT-base model to produce both a sen- tence representation, by applying mean-pooling, and an original contextualised sequence-length to- ken representation, which is used to initialise a RGCN.

Structure Information: We use Spacy depen- dency parser (Honnibal et al., 2020) with its middle model to obtain dependency parse trees for all input sentences. We

we found semantic graphs to be less effective than syntactic dependency trees

RGCNs, can be viewed as a weighted mes- sage passing process. At each RGCN layer, each node’s representation will be updated by collect- ing information from its neighbours and applying edge-specific weighting

. In our case, each
sentence is first parsed into a dependency tree, then modelled as a labelled directed graph by an RGCN, where nodes are words and edges are dependency relations.

we allow information to flow in both directions (from head to dependent and from dependent to head).

we pass BERT output through an embedding projection which is made of an affine transformation and ReLU non- linearity, then use the transformed representations to initialise RGCN’s node representations. Since

BERT and Spacy use different tokenisation strate- gies, we align them by taking the first subtoken as its word representation from BERT for each word in the RGCN.

We use a one-layer RGCN, as we find that a deeper network lowers the performance.

The concatenation of BERT and RGCN’s sentence representations are then passed through a layer normalisation layer to form the final sentence representation. Sentence embeddings of given sentence-pair are then inter- acted before passing to the final classifier for train- ing. In this siamese structure, all pa- rameters are shared and will be updated correspond- ingly. We use cross-entropy loss for optimisation

### CombOptNet: Fit the Right NP-Hard Problem by Learning Integer Programming Constraints - Paulus et al. 2021

Bridging logical and algorithmic reasoning with modern machine learning techniques is a funda- mental challenge with potentially transformative impact. On

On the algorithmic side, many **NP-HARD problems can be expressed as integer programs**, in which the constraints play the role of their “com- binatorial specification.” 

we aim to **integrate integer programming solvers into neural network architectures as layers capable of learning both the cost terms and the constraints**. 

end-to-end trainable architectures jointly extract features from raw data and solve a suitable (learned) combinatorial problem with state-of-the-art integer programming solvers

there are decades worth of research directed at solving complicated abstract tasks from their abstract formulation, it seems very difficult to align these methods with deep learning architectures needed for processing raw inputs. Deep

more structured paradigms that have more explicit reasoning
components, such as layers capable of convex optimiza- tion. In

we focus on **combinatorial optimiza- tion**, which has been well-studied and captures nontrivial reasoning capabilities over discrete objects; Enabling its unrestrained usage in machine learning models should fun- damentally enrich the set of available components

the main challenge of incorporat- ing combinatorial optimization into the model typically amounts to non-differentiability of methods that operate with discrete inputs or outputs.

3 basic approaches to overcome this are to 1

1. develop “soft” **continuous versions** of the discrete algorithms [44, 46]; 
2.  **adjust the topology** of neural network architectures to express certain algo- rithmic behaviour
3. provide an informative **gradient approximation** for the discrete algorithm

the last strategy requires nontrivial theoretical con- siderations, it can resolve the non-differentiability in the strongest possible sense; without any compromise on the performance of the original discrete algorithm. We follow this approach.

The most succesful generic approach to combinatorial op- timization is **integer linear programming (ILP).**

Integrating ILPs as building blocks of differentiable models is **chal- lenging because of the nontrivial dependency of the solu- tion on the cost terms and on the constraints.**

the **constraints** of an ILP are of critical interest due to their remarkable **expressive power**

Only **by modifying the constraints, one can formulate a number of diverse combinatorial problems** (SHORTEST-PATH,

, learning ILP constraints corresponds to learn- ing the combinatorial nature of the problem at hand.

we propose a backward pass (gradient compu- tation) for ILPs covering their full specification, allowing to use blackbox ILPs as combinatorial layers at any point in the architecture.

This layer can jointly learn the cost terms and the constraints of the integer program, and as such it aspires to achieve universal combinatorial expressivity

work that learns **how to solve combinatorial opti- mization** problems to improve upon traditional solvers that are otherwise computationally expensive or intractable,

**optimization** serves as a useful modeling paradigm to improve the applicability of machine learn- ing models and to add domain-specific structures and pri- ors. In

In the continuous setting, differentiating through op- timization problems is a foundational topic as it enables optimization algorithms to be used as a layer in end-to-end trainable models

This approach has been recently studied in the convex setting in OptNet [4] and Agrawal et al.

One use of this paradigm is to incorporate the knowledge of a downstream optimization-based task into a predictive model

Our goal is to incorporate an ILP as a differentiable layer in neural networks that inputs both constraints and objec- tive coefficients and outputs the corresponding ILP solu- tion.

we aim to embed ILPs in a blackbox man- ner: On the forward pass, we run the unmodified optimized solver, making no compromise on its performance. The task is to propose an informative gradient for the solver as it is. We never modify, relax, or soften the solver.

The task at hand is to provide gradients for the mapping (A, b, c) → y(A, b, c), in which the triple (A, b, c) is the specification of the ILP solver containing both the cost and the constraints

**main difficulty**: Since there are finitely many available  values of y, the mapping (A, b, c) → y(A, b, c) is piece-wise constant; and as such, its true gradient is zero almost everywhere

a small perturbation of the constraints or of the cost does typically not cause a change in the op- timal ILP solution. The zero gradient has to be suitably supplemented.

Gradient surrogates w.r.t. objective coefficients c have been studied intensively [see

Here, we focus on the differentiation w.r.t. constraints coefficients (A, b)

In the **LP** case, the in- tegrality constraint on Y is removed

* the optimal solution can be written as the unique solution to a linear system determined by the set of active constraints. 
* This captures the relationship between the constraint matrix and the optimal solution
* this relationship is differentiable.

in the case of an **ILP** the concept of active constraints vanishes. 

* There can be optimal solutions for which no constraint is tight. 
* Providing gradients for nonactive-but-relevant constraints is the principal difficulty. 
* The complexity of the interaction between the constraint set and the optimal solution is reflecting the NP-HARD nature of ILPs and is the reason why relying on the LP case is of little help.

*Method*

* we reformulate the gradient problem as a **descend direction task**. 
* We have to resolve an issue that the suggested gradient update y − dy to the optimal solution y typically unattainable, i.e. y − dy is not a feasible integer point
* Next, we generalize the concept of active constraints.  
* We substitute the binary information “active/nonactive” by a continuous proxy based on Euclidean distance

*Descent direction*. On the backward pass, the gradient of the layers following the ILP solver is given. Our aim is to propose a direction of change to the constraints and to the cost such that the solution of the updated ILP moves towards the negated incoming gradient’s direction (i.e. the descent direction).

The main advantage of this formulation is that it is meaningful even in the discrete case

every ILP solution y(A− dA, b − db, c − dc) is restricted to integer points and its ability to approach the point y − dy is limited unless dy is also an integer point.

*Constraints update*.

To get a meaningful update for a realizable change ∆k, we take a gradient of a piecewise affine local mismatch function P∆k
.The definition of P∆k
is based on a geometric understanding of the underlying structure

t tighter constraints con- tribute more to P∆k
. In this sense, the mismatch function
generalizes the concept of **active constraints**.

 In prac- tice, the minimum is softened to allow multiple constraints to be updated simultaneously. For

our mapping dy ?→ dA, db is homogeneous. It is due to the fact that the whole situation is rescaled to one case (choice of basis) where the gradient is computed and then rescaled back (scalars λk). The

*Constraint parametrization.* For learning constraints, we have to specify their parametrization. The represen- tation is of great importance, as it determines how the constraints respond to incoming gradients. Additionally, it affects the meaning of constraint distance by changing the parameter space.

We represent each constraint (ak, bk) as a **hyperplane** de- scribed by its normal vector ak, distance from the origin rk and offset ok of the origin in the global coordinate system

Compared to the plain parametrization which represents the constraints as a matrix A and a vector b, our slightly **overparametrized** choice allows the constraints to rotate without requiring to traverse large distance in parame- ter space

KNAPSACK example

a KNAPSACK instance consists of 10 sentences, each describing one item. The sentences are preprocessed via the sentence embedding [12] and the 10 resulting 4 096-dimensional vectors x con- stitute the input of the dataset. We rely on the ability of natural language embedding models to capture numerical values, as the other words in the sentence are uncorrelated with them

The indicator vector y∗ of the optimal solution (i.e. item se- lection) to a knapsack instance is its corresponding label (Fig. 10). The dataset contains 4 500 training and 500 test pairs (x, y∗)

We propose a method for **integrating integer linear pro- gram solvers into neural network architectures as layers.**

enabled by **providing gradients for both the cost terms and the constraints** of an ILP

The resulting end- to-end trainable architectures are able to simultaneously extract features from raw data and learn a suitable set of constraints that specify the combinatorial problem

it strives to achieve universal combinatorial expressivity in deep networks – opening many exciting perspectives

### An Integer Linear Programming Framework for Mining Constraints from Data - Meng and Chang 2021

Structured output prediction problems (e.g., se-
quential tagging, hierarchical multi-class classi-
fication) often involve constraints over the out-
put label space. These constraints interact with
the learned models to filter infeasible solutions
and facilitate in building an accountable system.
However, although constraints are useful, they are
often based on hand-crafted rules. This raises a
question – can we mine constraints and rules from
data based on a learning algorithm?

For example, in part-of-speech
tagging, a constraint specifying that every sentence should
contain at least one verb and one noun can greatly improve
the performance (Ganchev et al., 2010). Similarly, in hier-
archical multi-label classification, a figure labeled ‘flower’
should also be labeled ‘plant’ as well (Dimitrovski et al.,
2011). To incorporate constraints with learned models, one
popular method is to formulate the inference problem into
an integer linear programming (**ILP**) (Roth & Yih, 2004).
This framework is general and can **cope with constraints**
**formed as propositional logics**

**ILP is a linear optimization problem with linear constraints**
**and the values of variables are restricted to integers**



### Hybrid self-interactive attentive siamese network for medical textual semantic similarity - An et al. 2020

medical texts, but it is difficult to process these texts due to many similar sentences

estimating the similarity of medical texts has become a key technology, filtering out medical texts quickly. Nowadays,

many methods for estimation similarity between medical sentences extract semantic features mainly via Siamese network

these methods don't achieve the best results due to the large amount of noise in the texts.

To improve the performance of the Siamese network, a hybrid self-interactive attention model is proposed in this paper. The

The aim is to reduce the noise of the text and strengthen the token with high correlation between the two texts

uses BERT as the embedding layer to carry out a preliminary pre- training of text.

it is difficult to obtain interesting texts from these medical texts due to the similarity among these sentences.

The prediction of textual semantic similarity can solve this problem.

medical texts often contain professional and uncommon medical words with similar semantics, which makes it difficult to evaluate the similarity of medical texts from the perspective of words and sentences. Siamese

**Siamese convolution network solve the task of text similarity**

The Siamese network contains **two inputs**, which correspond to two texts to predict similarity. The Siamese network extracts the semantic features of the two input sentences through **two sub-networks with shared parameters** and evaluates the similarity between sentences via the extracted semantic features.

the sub-network in Siamese network can be replaced. In Siamese network[7], recurrent neural network is used as sub-network

Siamese network is better than information retrieval and word frequency co-occurrence in text similarity, there is still the possibility to continue to improve performance via integrating other existed method/mechanism.

**Attention mechanism** can reprocess the two semantic features of Siamese network, the relevant part of semantic feature is enlarged and the noise is reduced, which may improve the evaluation results

Attention mechanism can be divided into 

1. **self-attention** mechanism
   * can expand the useful part of the text and reduce the useless part, so that the model can quickly learn useful features. Self-attention mechanism is mainly used in text **categorization**
2.  **interactive** attention mechanism 
   *  application of **text similarity** task in attention mainly uses interactive attention mechanism to improve the semantic representation between two sentences

interactive attention is more frequently used in text similarity tasks, because interactive attention mechanism can calculate a comprehensive attention weight via two input semantic features

**The interactive attention weights denote the correlation between the two semantic features**

Nevertheless, the interactive attention mechanism cannot play any role in two unrelated sentences, and it may even magnify the unrelated parts of the features. Such a model often predicts higher results than the real value. For

we modify the interactive attention mechanism by multiplying each sentence feature by the self-attention weight of the comparative sentence feature, which not only learns the features of the comparative sentence, but also makes the text independent.

a fatal weakness of interactive attention in text similarity task that it cannot enhance the independence of each sentence while improving the interaction between sentences.

this happens to be the advantage of self-attention mechanism

Self-attention mechanism strengthens the more influential part of its own text by learning its own text.

Considering from the perspective of text similarity task, it makes sentences more independent

**similarity score** of text similarity task prediction is the distance calculated by distance formula according to the semantic features trained by Siamese network

to increase the interaction between the two texts and keep the independence of the text from declining, we try to calculate the distance between the semantic features obtained by self-attention and the semantic features obtained by interactive attention (so we can get two distances), and then calculate the average of the two semantic distances to get the final prediction results.

BERT pre-training sentence vectors are used as input, and the initial semantic features of two sentences are obtained by training the vector representation of two sentences through Bi-LSTM network of Siamese framework

(BERT) as the embedding layer of the experiment. BERT[11]implements context dependency in text by bidirectional Transformer encoding. Transformer[8]is the network structure of the encoder-decoder framework. All tokens in the sequence can be processed in parallel due to multi-head attention mechanism.

the Bi-directional Long Short-Term Memory (Bi-LSTM) network ,performing well in the RNN ,is chosen as our sub-networks. Bi-LSTM consists of forward LSTM and backward LSTM. Meanwhile, LSTM model contains forgetting gate, memory gate and output gate. The model discards useless information through forgetting gate, stores useful information through memory gate, and finally outputs the results via output gate.

Hybrid self-interactive attention uses both self-attention and interactive attention to integrate their characteristics

text similarity experiments need to compare the similarity between two texts, we need to pay attention not only to each text itself, but also to the degree of influence between texts. In our experiment, we used three kinds of Interactive Attention. Interactive Attention Network, Single Interactive Attention and Merge Interactive Attention

### Differentiation of blackbox combinatorial solvers - Vlastelica et al. 2020

fusion of deep learning with combinatorial algorithms promises

One possible approach is to introduce combinatorial building blocks into neural networks.

Such end-to-end architec- tures have the potential to tackle combinatorial problems on raw input data such as ensuring global consistency in multi-object tracking or route planning on maps in robotics.

a method that implements an efficient back- ward pass through blackbox implementations of combinatorial solvers with linear objective functions

toolbox of popular methods in computer science currently sees a split into two major com- ponents

1. classical algorithmic techniques from discrete optimization – graph algorithms, SAT-solvers, integer programming solvers – often with heavily optimized im-
   plementations and theoretical guarantees on runtime and performance

2. deep learning allowing data-driven feature extraction as well as the flexible design of end-to-end architectures

The **fusion of deep learning with combinatorial optimization** is desirable both for foundational reasons – extending the reach of deep learning to data with large combinato- rial complexity – and in practical applications.

for example in computer vision problems that require solving a combinatorial sub-task on top of features extracted from raw input such as establishing global consistency in multi-object tracking from a sequence of frames

**The fundamental problem with constructing hybrid architectures is differentiability of the combina- torial components.**

State-of-the-art approaches pursue the following paradigm: introduce suitable **approximations or modifications of the objective function** or of a baseline algorithm that eventu- ally yield a differentiable computation. 

* The resulting algorithms are often sub-optimal in terms of runtime, performance and optimality guarantees when compared to their unmodified counterparts.

the sources of sub-optimality vary from example to example, there is a common theme: **any differentiable algorithm in particular outputs continuous values and as such it solves a relaxation of the original problem**. It

It is well-known in combinatorial optimization theory that even strong and practical convex relaxations induce lower bounds on the approximation ratio for large classes of problems which makes them inherently sub-optimal

a method that, **at the cost of one hyperparameter, implements a backward pass for a blackbox implementation of a combinatorial algorithm or a solver that optimizes a linear objective function**

This effectively **turns the algorithm or solver into a composable building block of neural network architectures,** as

The main technical challenge boils down to **providing an informative gradient of a piecewise con- stant function**.

leverage the minimization structure of the un- derlying combinatorial problem and efficiently compute a gradient of a continuous interpolation

The computational cost of the introduced backward pass matches the cost of the forward pass.

The task to solve during back-propagation is the following. We receive the gradient dL/dy of the global loss L with respect to solver output y at a given point ˆy = y( ˆw). We are expected to return dL/dw, the gradient of the loss with respect to solver input w at a point ˆ

Since Y is finite, there are only finitely many values of y(w). In other words, this function of w is piecewise constant and the gradient is identically zero or does not exist (at points of jumps).

* if one does a small perturbation to edge weights of a graph, one usually does not change the optimal TSP tour and on rare occasions alters it drastically. This

**The fundamental problem with differentiating through combinatorial solvers is not the lack of differentiability; the gradient exists almost everywhere. However, this gradient is a constant zero and as such is unhelpful for optimization.**

we will not rely on standard techniques for gradient estimation



### Optimization with Constraint Learning: A Framework and Survey - Fajemisin et al. 2021

In ML, a loss function provides a measure of the difference between the value predicted by the ML algorithm and the actual value. It follows then that ML practitioners seek to minimize this loss function, which leads to the use of **Optimization** for ML

In certain applications, e.g. deep neural networks, the objective function of the learning task is non-convex, leading to the application of non-convex op- timization methods

machine learning is inherently linked to optimization,

more and more attention for using ML
for optimization. Optimization

In addition to using traditional ML approaches to learn constraints for optimization problems (e.g. regression trees and linear regression in (Verwer et al., 2017) and neural networks, other tech- niques such as genetic programming , mixed-integer linear programming (MILP) and symbolic regression have also been used.

### Machine learning for combinatorial optimization: A methodological tour d'horizon - Bengio et al. 2021

leveraging machine learning to solve combinatorial optimization problems.

pushing further the integration of machine learning and combinatorial optimization and

seeing generic optimization problems as data points and inquiring what is the relevant distribution of problems to use for learning on a given task.

we focus on **discrete optimization problems** formulated as **integer constrained optimiza- tion,** i.e. , with integral or binary variables (called decision vari- ables). While

not all such problems are hard to solve ( e.g. , shortest path problems), we concentrate on **combinatorial optimization problems** (**NP-hard**)

for those problems, it is considered unlikely that an algorithm whose run- ning time is polynomial in the size of the input exists.

in practice, combinatorial optimization algorithms can solve instances with up to millions of decision variables and constraints

diverse types of problems are solved by leveraging their special structure.

Other algorithms, designed to be general, are found in hindsight to be empirically more efficient on particular problems types

The focus of this paper is on combina- torial optimization algorithms that automatically perform learning on a chosen implicit distribution of problems. Incorporating ma- chine learning components in the algorithm can achieve this.	

machine learning focuses on performing a task
given some (finite and usually noisy) data

It is well suited for nat- ural signals for which no clear mathematical formulation emerges because the true data distribution is not known analytically, such

Most of the times, the learning problem has a statistical formulation that is solved through mathematical optimization.

dramatic progress has been achieved with deep learning, an machine learn- ing sub-field building large parametric approximators by compos- ing simpler functions. Deep

From the combinatorial optimization point of view, machine
learning can help improve an algorithm on a distribution of prob- lem instances in two ways. On

1. On the one side, the researcher as- sumes expert knowledge 2 about the optimization algorithm, but wants to replace some heavy computations by a fast approxi- mation. 
   * **Learning can be used to build such approximations in a generic way**, i.e. , without the need to derive new explicit algo- rithms.

2. expert knowledge may not be sufficient and some algorithmic decisions may be unsatisfactory. The goal is therefore to **explore the space of these decisions, and learn out of this experience the best performing behavior** (policy), hopefully improving on the state of the art. Even

Even though machine learning is approximate, we will demonstrate through the examples surveyed in this paper that this does not systematically mean that incor- porating learning will compromise overall theoretical guarantees

The challenge in learning is that an algorithm that per- forms well on problem instances used for learning may not work properly on other instances from the true probability distribution

To control this, we monitor the performance of the learned algorithm over another independent set of unseen problem instances. Keeping the perfor- mances similar between the instances used for learning and the unseen ones is known in machine learning as generalizing . Cur-

Without loss of generality, **a combinatorial optimization problem can be formulated as a constrained min-optimization program**

* **Constraints** model natural or imposed restrictions of the problem 
* **variables** define the decisions to be made
* the **objective func- tion**, generally a cost to be minimized, defines the measure of the quality of every feasible assignment of values to variables

If the objective and constraints are linear, the problem is called a **lin- ear programming** problem. 

If, in addition, some variables are also restricted to only assume integer values, then the problem is a **mixed-integer linear programming** problem

The set of points that satisfy the constraints is the **feasible region.** 

* Every point in that set (often referred to as a **feasible solu- tion) yields an upper bound on the objective value of the optimal solution**. Exact

* Exact solving is an important aspect of the field, hence a lot of attention is also given to find **lower bounds** to the optimal cost.

* The tighter the lower bounds, with respect to the optimal so- lution value, the higher the chances that the current algorithmic approaches to tackle mixed-integer linear programming described in the following could be successful, i.e. , effective if not effici

**Linear and mixed-integer linear programming** problems are the
workhorse of **combinatorial optimization** because **they can model a wide variety of problems and are the best understood**, i.e. , there are reliable algorithms and software tools to solve them.

With respect to complexity and solution methods, **linear programming** is a **poly- nomial problem**, well solved, in theory and in practice, through the **simplex algorithm or interior points methods**. 

**Mixed-integer linear programming**, on the other hand, is an **NP-hard** problem, which does not make it hopeless. Indeed, it is easy to see that the com- plexity of mixed-integer linear programming is associated with the integrality requirement on (some of) the variables, which makes the mixed-integer linear programming feasible region nonconvex

the algorithmic line of attack that is used to solve mixed-integer linear programming through a whole ecosystem of branch-and-bound techniques to perform implicit enumeration

All commercial and noncommercial mixed-integer linear pro-
gramming solvers enhance the above enumeration framework with the extensive use of **cutting planes**, i.e. , valid linear inequalities that are added to the original formulation (especially at the root of the branch-and-bound tree) in the attempt of strengthening its lin- ear programming relaxation

In **supervised learning**, a set of input (fea-
tures) / target pairs is provided and the task is to find a function that for every input has a predicted output as close as possible to the provided target. Finding such a function is called **learning** and is solved through an **optimization** problem over a family of func- tions. 

* The **loss function**, i.e. , the measure of discrepancy between the output and the target, can be chosen depending on the task (regression, classification, etc. ) and on the optimization methods. 
* this approach is not enough because the problem has a statistical nature. It is usually easy enough to achieve a good score on the given examples but one wants to achieve a good score on unseen examples (test data). This is known as generalization.

The loss function ? is task dependent ( e.g. , classifica- tion error) and can sometimes be replaced by a surrogate one ( e.g. , a differentiable one). The

The probability distribution is unknown and inaccessible. , it is approximated by the empirical probability distribution over a finite training dataset and the optimization problem is solved

If a model ( i.e. , a family of functions) can represent many differ- ent functions, the model is said to have high capacity and is prone to overfitting: doing well on the training data but not generaliz- ing to the test data.

**Regularization** is anything that can improve the test score at the expense of the training score and is used to restrict the practical capacity of a model. On

if the capacity is too low, the model underfits and performs poorly on both sets. The

he boundary between overfitting and underfitting can be estimated by **changing the effective capacity (the richness of the family of functions reachable by training):** below the critical capacity one underfits and test error decreases with increases in capacity, while above that critical capacity one overfits and test er- ror

**Selecting the best among various trained models cannot be**
**done on the test set. Selection is a form of optimization, and do- ing so on the test set would bias the estimator in**

To perform **model selection**, a validation dataset D v alid is used to es- timate the generalization error of different machine learning mod- els is necessary.

Model selection can be done based on these esti- mates, and the final unbiased generalization error of the selected model can be computed on the test set. The

The validation set is there- fore often used to select effective capacity, e.g. , by changing the amount of training, the number of parameters θ, and the amount of regularization imposed to the model.

**unsupervised learning** has received so far little attention in conjunction with combinatorial optimization and its immediate use seems difficult, we are not discussing it any further

In **reinforcement learning**, an agent interacts with an environ-
ment through a markov decision process,

At every time step, the agent is in a given state of the environment and chooses an action according to its (possibly stochastic) policy. As a result, it receives from the environment a reward and enters a new state. T

The goal in reinforcement learning is to train the agent to maximize the expected sum of future rewards, called the return

For a given policy, the expected return given a current state (resp. state and action pair) is known as the **value function** (resp. state action value function). 

Value functions follow the Bellman equa- tion, hence the problem can be **formulated as dynamic programming**, and **solved approximately**. The

The **dynamics of the environment** need not be known by the agent and are learned directly or in- directly, yielding an **exploration vs exploitation dilemma**: choos- ing between exploring new states for refining the knowledge of the environment for possible long-term improvements, or exploit- ing the best-known scenario learned so far (which tends to be in already visited or predictable states).

the simplest neural network architecture, the feedforward neural network (also called an multilayer perceptron),

For every layer, an **affine transformation** is applied on the input vector, followed by a **non-linear scalar function** (named **activation function**) applied element-wise. The

The output of a layer, called **intermediate activa- tions**, is passed on to the next layer

All affine transformations are independent and represented in practice as different **matrices of coefficient**

They are learned, i.e. , **optimized** over, through stochastic gradient descent, the optimization algorithm used to minimize the selected loss function.

The **stochasticity** comes from the **limited number of data points used to compute the loss** before applying a gradient update. In

In practice, gradients are computed using **reverse mode automatic differentiation**, a practical algorithm based on the chain rule, also known as **back-propagation**. Deep

Deep neural networks can be difficult to optimize, and a large variety of techniques have been developed to make the optimization behave better, often by changing architectural designs of the network

have dramatic capacities, i.e. , they can essentially match any dataset, thus being prone to overfitting, they are also heavily regularized. Training

Training them by stochastic gradient descent also regularizes them because of the noise in the gradient, making neural networks generally robust to over- fitting issues, even when they are very large and would overfit if trained with more aggressive optimization. In

many hyper-parameters exist and different combinations are evaluated (known as hyper-parameters optimization). Deep

Deep learning also sets itself apart from more traditional machine learning techniques **by taking as inputs all available raw features of the data**, e.g. , all pixels of an image, while traditional machine learning typi- cally requires to engineer a limited number of domain-specific features.

**recurrent neural network**. These models can operate on sequence data by sharing parameters across different sequence steps. 

* a same neural network block is successively applied at every step of the sequence, i.e. , with the same architecture and parameter values at each time step.

* specificity of such a network is the presence of **recurrent layers**: layers that take as input both the activation vector of the previous layer and its own activation vector on the preceding sequence step (called a hidden state vector), as

 **attention mechanisms .**

* size-invariant technique 
*  can be used to process data where each data point corresponds to a set. 
* parameter sharing is used to address the fact that different sets need not to be of the same size
*  Attention is used to query information about all elements in the set, and merge it for downstream processing in the neural net- work
*  An **affinity function** takes as input the **query** (which represents any kind of **contextual information which informs where attention should be concentrated**) and a **represen- tation** of an element of the set (both are activation vectors) and outputs a **scalar**. 
* This is repeated over all elements in the set for the same query. Those scalars are normalized (for instance with a softmax function) and used to define a weighted sum of the representations of elements in the set that can, in turn, be used in the neural network making the query. T

* Attention can be used to build **graph neural network**, i.e. , neural networks able to process graph structured input data, 
  * every node attends over the set of its neighbors.
  *  The process is repeated multiple times to gather information about nodes further away. graph neural network can also be understood as a form of message passing (
