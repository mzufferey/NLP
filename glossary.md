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