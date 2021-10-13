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