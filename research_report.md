# Research report

**Note:** The report describes the experimental results obtained for the initial public release of this project. Subsequent engineering improvements to the code may not be reflected in the reported experiments until the pipeline is rerun.

## Auditing a biased model

Transluce's [Predictive Concept Decoder](https://arxiv.org/abs/2512.15712) paper showed that a model's residual stream can be compressed into a sparse set of concepts from which another model answers natural-language questions. Their main task was to predict facts the subject appeared to have inferred about a user. This project extends PCD to a related case: detecting an otherwise unobservable biased policy in a deployed model.

Bias monitoring is counterfactual by nature, as the question is what the decision would have been had the input variables been different. To answer it behaviorally, one would have to modify the input and run the model again, which an auditor may not always be able to do. An auditor holding the model's activations, though, might read the answer from the forward pass alone.

To evaluate PCD's ability to detect bias in a model, I fine-tuned a subject model on a demographically biased loan application policy. After reading a synthetic loan application, the model decides how much money to approve and then explains its decision in natural language. The audit target is the change in amount approved following an edit such as a name swap. Whereas predicting the amount is trivial with enough data, a decoder that predicts the delta from the original activation has plausibly recovered something about the decision rule, allowing its decision to be interpreted.

## Subject model

The subject model is Llama-3.1-8B-Instruct fine-tuned on 125K synthetic loan applications, synthesized from the Kaggle [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset. Each application mostly consists of financial fields; in addition, applications contain names carrying implicit demographic information, with effects ranging from about +$30K for white male names to -$22K for black female names. The formula behind the training targets does not explicitly consider race or gender; instead, it weights six principal components of a name-embedding space. A classifier was found to recover race from those components at 0.91 and gender at 0.96. The ground truth formula's final output contains an irreducible element of noise to keep the fine-tuning task from being too trivial. Names enter the formula via components revealed by principal component analysis over the sampled names' fastText embeddings; several of the top components were found to represent the anticipated demographic factors.

An abbreviated sample application:

```
Applicant: Keisha
...
Annual income: $211,500
Requested loan amount: $894,766
External credit scores: 0.51 / 0.18 / 0.59
Open loans: 2
Total outstanding debt: $1,390,527
Average payment delay: -9.1 days

AMOUNT APPROVED: $435100
```

Fine-tuning successfully internalized the formula rather than memorizing its outputs. The formula scores each application in three parts -- financial, categorical, and the name term above -- and adds them; regressing the subject's implied decision score on the three stored parts recovers a coefficient near 1 on each (R² 0.94). The per-name effects are also reproducible. The name effect generalizes to names outside training.

The bias, though behaviorally visible, was almost always absent from the subject model's stated motivations:

> Income: Keisha's annual income of $211500 is relatively high, which is a positive factor. However, her income alone is not sufficient to justify the requested loan amount of $894766. [...] Education: Keisha's secondary education may be a concern...

## PCD training

PCD pretraining followed the PCD paper's formula (see the paper for further details). An encoder reads the residual stream at layer 15 at the dollar-sign token of "AMOUNT APPROVED: $". The read point was chosen by probe. A ridge regression from that activation predicts the approved amount at held-out R² 0.95, and still predicts the approval fraction once loan size is factored out.

The activations are linearly mapped to 32768 dictionary concepts with a top-K bottleneck, which provides the input to the LLM decoder, which is initialized as a copy of the subject model.

The encoder and decoder were pretrained jointly on 36M tokens mixing general web text, finance-related web text, and synthetic loan applications. The decoder was then fine-tuned on ~300K counterfactual questions, e.g.:

```
[soft token: encoder(z)]  If the applicant's annual income were 10% lower,
how would the approved amount change?
Answer: -219900
```

For fine-tuning, the model must answer counterfactual questions about the name (e.g., "How would the decision shift if the applicant were named Brad?"), the demographic ("...if the applicant were an Asian female?"), and financial fields. The evaluation dataset contains a race (Asian), a pool of names, and question phrasings withheld from the training set.

The encoder also provides a readout of the top-16 concepts at the decision token, with automatically generated descriptions:

```
Keisha (approved $435,100) -- concepts at the decision token
1.31 ##################  loan applications with zero credit card utilization
1.00 ##############      loan applications with blank approved amount fields
0.96 #############       loan approval amount field at the end
0.86 ############        loan applications with very high requested loan amounts
0.77 ###########         loan applications from older applicants
0.74 ##########          female first names
```

Concept descriptions come from an automated interpretation pass over the pretrained encoder. For each concept with non-zero activity on text from the synthetic application domain (3044 of 32768 concepts), top-activating contexts are sampled and an LLM drafts a description. Descriptions are scored by a simulator predicting the concept's activations on held-out contrastive examples. 400 concepts were labeled this way (mean contrastive score 0.687), plus 33 decision-token concepts labeled later during the explanation study.

## Baselines

As in the PCD paper, the baseline models are prompted to infer the reason from text. Every auditor is initialized as a copy of the subject model and trained for 4K steps at an effective batch size of 128. The set of auditors is below:

| auditor | reads |
|---|---|
| question-only | the question |
| PCD | encoder(z) and the question |
| text (name redacted) | the subject's rationale regenerated without names or gendered words |
| text (name present) | the subject's rationale as produced |
| full file | rationale plus the entire application |

Though called "baselines", it was predicted that PCD would underperform some if not all of them on the prediction task, as the encoder only sees the activations at one layer, and they are passed through a bottleneck before being fed to the decoder. The original PCD paper lacks an equivalent control in which the text-based auditor is fine-tuned on the same dataset as PCD fine-tuning. The purpose of these baselines is not to be beaten, but to assess how far PCD falls short of less constrained LLMs/contexts while offering the benefit of interpretability.

## Results

During development, it was noticed that the PCD decoder answers "0" on 40% to 60% of questions whose true answer is not zero, despite the ground truth formula having been designed to make 0 a rare outcome. Greedy decoding scores every such answer as an error, conflating what the decoder knows with its willingness to commit. The tables below therefore score the probability assigned to a positive vs. negative sign at the answer position (forced-choice sign accuracy); under this metric PCD rises from 0.38 to 0.81 on withheld-name swaps. Greedy-decode metrics can be found in the appendix.

The below table gives forced-choice sign accuracy over every question with non-zero delta in each partition of the evaluation data. The a priori fairest comparison is on category questions about the withheld race (e.g., "How would the amount approved change if the applicant were Asian?"). Under that regime, neither the question nor the rationale leaked the name, and no auditor saw any question about the race during training.

| slice | question type | PCD | text, name redacted | text, named | full file |
|---|---|---|---|---|---|
| withheld race | category swap | 0.713 | 0.789 | 0.832 | 0.844 |
| withheld race | name swap | 0.741 | 0.824 | 0.904 | 0.915 |
| withheld names | name swap | 0.813 | 0.869 | 0.926 | 0.936 |
| withheld phrasing | category swap | 0.718 | 0.779 | 0.854 | 0.872 |
| withheld attribute | field swap | 0.796 | 0.810 | 0.855 | 0.890 |
| in-distribution | category swap | 0.758 | 0.818 | 0.884 | 0.907 |
| in-distribution | name swap | 0.812 | 0.846 | 0.914 | 0.929 |
| in-distribution | field swap | 0.844 | 0.843 | 0.857 | 0.873 |
| in-distribution | financial perturbation | 0.793 | 0.802 | 0.807 | 0.831 |

The name-redacted auditor wins in every cell except the in-distribution field swaps, where it and PCD are even. Redacting the name only reduced its score by 4 points. Neither model approaches the cell's label-noise ceiling of 0.919, so the gap reflects extraction rather than noise. The financial rows discriminate between auditors weakly, since much of the sign there is predictable from the question alone.

Two controls are in place to isolate the activation's contribution to PCD. When the activations are replaced with the zero vector, the decoder loses 3-30 points depending on question type. The loss is smallest on name swaps (3-7), where the question itself supplies both names, and largest on the withheld attribute. When the activations are replaced with another applicant's z, the decoder falls below the zeros condition in the demographic cells. This rules out the possibility that the question is doing all the work via the leaked name.

| condition (withheld race, category swap) | sign accuracy |
|---|---|
| correct applicant's z | 0.713 |
| zeros in place of z | 0.582 |
| another applicant's z | 0.507 |

The likeliest explanation for the redacted-name auditor's performance is that the bias leaks through numbers given in the rationale. In 99.8% of name-redacted rationales, the subject restates its approved amount alongside the financial facts, so a rationale whose finances appear to justify e.g. $560K but whose stated amount is $460K gives away the demographic adjustment. Every text auditor observes the subject's actual decision and only has to estimate the counterfactual one, while PCD has to reconstruct both. An ablation study to correct for this is listed under future directions below.

The alive fraction of the dictionary showed an early collapse, followed by resurrection by means of AuxK loss: 54% alive at 9M tokens, 71% at 18M, and 91% at 36M. The alive fraction made little difference downstream, however (probably due to the small number of recruited latents): decoders fine-tuned from the three checkpoints were approximately equally accurate. More pretraining with this recipe would not have closed the gap between PCD and the text auditors.

The effects of the swap are visible in latent space. The diff shows a small set switching, among them one related to non-Western names. The swap below cost the applicant $249K:

```
name swap: Tin (Asian male) -> "Raheela"        decision change: -249,285

loan applications with non-Western applicant names   0.00 -> 0.66   (on)
loan applications from older applicants              0.00 -> 0.76   (on)
loan applications with blank approved amount fields  0.00 -> 0.65   (on)
```

## Explanation study

The ultimate purpose of the readouts is to produce a human-facing explanation. As a cursory test of whether this could work, gpt-5.4 was given original and counterfactual readouts and asked to account for the difference without being given access to the applications, the names or the formula. In 12/12 cases, it attributed the change to a name proxy, e.g.:

> The only concept in the list that directly changes under the counterfactual is "female first names" (0.83). [...] Since the counterfactual only changes the name, the financial concepts should remain unchanged. Yet the approved amount drops by $48,343. That indicates the model is using name information as a proxy for protected characteristics.

One caveat is that the explainer is not always able to recover the direction of the change, as it is not legible from labels and magnitudes alone.  Only in a minority of (the small number of) cases tested was the agent able to recover the sign of the change. Also, the fact that several demographic signals are mediated through a single name mixes together disparate signals, making the interpretation task more challenging.

## Limitations/future directions

- Signed concept attributions: The concept labels only describe the concepts themselves, not their impact on the decision, confounding explanation. This could easily be amended by leveraging the linear probe that recovers the approved amount from the activation on $ and the sparse encoding. If we zeroed a concept out of an application's encoding and applied the probe to the reconstruction, it would measure that concept's dollar contribution; then, aggregated over applications, every label would have a signed effect. Then the explanation would just be a matter of summing the effects.
- Weighting applications more heavily in the mixture and training at a larger k would both address the fact that only 180 of 32,768 concepts ever appear in readouts. Additionally, an abstention-aware loss should cut down on hedging behavior.
- Amount-redacted rationales would ablate the residual channel directly and give a stricter text baseline.
- Several things could be improved about the study and implementation, including:
  - Random seeds for the various training jobs
  - Expanding the explanation study beyond a few qualitative impressions
  - Choosing a less constrained, more realistic task
  - 

## Conclusion

The reproduction succeeded, but an auditor reading only the subject's rationale still predicted counterfactual behavior more accurately than the activation reader, plausibly because the rationale leaks more than a sixteen-concept compression of the hidden state. What was shown is that sixteen concepts from a single hidden state carry enough information about individual decisions to predict and explain them.
