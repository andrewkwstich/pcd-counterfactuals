# Appendix: full results


## A. Subject model

Llama-3.1-8B-Instruct + LoRA (r=16), fine-tuned on 125K synthetic applications rendered from a formula with injected demographic effects routed through the name. Validation: amount parses in 100% of decodes; subject-vs-formula MAE $42.8K; a linear probe on the layer-15 activation at the decision token recovers the amount at R² 0.95 (the read point used throughout).

Measured mean decision change under name swap, by target cell (80,000 swaps):

| cell | mean delta |
|---|---|
| white male | +29.7K |
| hispanic male | +9.1K |
| white female | +6.7K |
| black male | +4.3K |
| asian male | +4.2K |
| hispanic female | -11.1K |
| asian female | -18.3K |
| black female | -22.0K |

##. Validation gates

Three checks had to pass before anything downstream was built.

| gate | question | criterion | result |
|---|---|---|---|
| proxy validation | are race and gender recoverable from the name features the formula uses? | well above chance | race 0.91, gender 0.96 (5-fold CV, balanced n=4000) |
| localization | does the activation at the read point carry the decision? | R² ≥ 0.80 | 0.951 |
| subject validation | did the subject learn the formula rather than memorize outputs? | four checks below | pass |

Proxy validation. The decision formula receives no explicit race or gender variable at inference time; its name-component weights were calibrated to create specified demographic effects through the PCA representation. It weights six principal components of a name-embedding space, and the gate confirms a classifier can recover race and gender from those six components, so the name is a working proxy.

Localization. A ridge regression from the layer-15 activation at the decision token predicts the approved amount at held-out R² 0.951. Because raw amount is confounded with loan size, the probe also predicts the approval fraction, a scale-free target, at 0.823; the activation carries the decision and not merely the size of the loan under discussion.

Subject validation, four checks. First, outputs parse cleanly (100% of decodes yield an amount). Second, the subject learned the formula: its decisions imply an approval fraction for each application, and regressing that implied fraction on the formula's three ingredients (financial score, categorical score, name score) recovers coefficients of 1.06, 1.11, and 1.04 with R² 0.936, where learning the formula exactly would give coefficients of 1. Third, the name's influence is unambiguous (partial F = 9776, p ≈ 0), and it flows through the same name component the formula weights most heavily, which is what one would expect if the subject learned the name-space function that generalizes to unseen names instead of a lookup table over trained names. Fourth, the name-swap deltas are stable: computing them twice on disjoint halves of the data gives correlated results at r = 0.998, so the bias is a reliable property of the model.

Later checks in the same spirit: zero leakage of any held-out slice into auditor training (verified at dataset build), and zero lexical leaks in the regenerated name-free rationales (section D).

## B. Counterfactual QA dataset

372,874 questions over 20,000 applications; labels from re-running the frozen subject on the edited application.

| class | count | zero-delta rate |
|---|---|---|
| financial perturbation (quant) | 160,000 | 41% |
| categorical field swap | 92,874 | 35% |
| name swap (name_point) | 80,000 | 25% |
| category swap (name_categorical, mean over 16 sampled names) | 40,000 | 7% |

| split | n | meaning |
|---|---|---|
| pcd_train | 280,759 | auditor training |
| sense1_test | 31,226 | in-distribution test, application-level holdout |
| sense2_race | 29,755 | all questions targeting the withheld race (asian) |
| sense2_attribute | 20,000 | withheld attribute (own_home) |
| sense2_names | 6,080 | withheld name pool |
| sense2_phrasing | 5,054 | withheld phrasing |

Zero leakage of any sense2 slice into training (verified at build time). Sign metrics below use nonzero-delta questions; zero-delta questions test predicting insensitivity and are excluded from those tables.

## C. PCD pretraining

Paper-faithful: m=32,768, k=16, read layer 15, write layer 0, aux loss eps=1e-4 / k_aux=500 / 1M-token dead window, AdamW 1e-4, wd 0.01, cosine to 0, no warmup, effective batch 128. Mixture 72% FineWeb, 18% finance-filtered FineWeb, 10% applications (decision line masked from the loss). 36M encoder tokens, 17,579 steps, 3.1 h on 4xH100.

| checkpoint | tokens | dictionary alive |
|---|---|---|
| ckpt_9M | 9.0M | 0.538 |
| ckpt_18M | 18.0M | 0.712 |
| final | 36.0M | 0.908 |

Final smoothed LM loss 2.60. Auto-interp over the final encoder: 3,044 concepts live on loan-domain text, 400 labeled, mean contrastive simulator score 0.687; 33 additional decision-token concepts labeled during the explanation study (labels in `artifacts/explanations/concept_labels_merged.json`).

## D. Auditor fine-tunes

All auditors: same merged subject base + fresh LoRA (r=16), 4,000 steps at effective batch 128, cosine to 0, identical targets and splits. PCD continues the pretraining LoRA with the encoder frozen.

| run | input | val CE | val exact match | wall clock |
|---|---|---|---|---|
| pcd-finetune-v1 | encoder(z) + question | 1.298 | 0.272 | 0.7 h |
| pcd-finetune-9M / -18M | scale variants | 1.305 / 1.308 | 0.271 / 0.273 | 0.8 h each |
| baseline-f1prime-v1 | name-free rationale + question | 0.869 | 0.384 | 1.9 h |
| baseline-f1-v1 | named rationale + question | 0.834 | 0.394 | 2.6 h |
| baseline-f2-v1 | rationale + application + question | 0.767 | 0.433 | 3.5 h |

Name-free rationales: re-elicited from the frozen subject with a no-name / no-gendered-language instruction; lexical gate (names, pronouns, race and gender terms) passes 19,994/20,000 on the first generation, 4 on retry, 2 by deterministic scrub; zero residual leaks.

## E. Forced-choice sign accuracy, full N (headline metric)

Probability of " +" vs " -" at the answer position; every nonzero-delta question in each slice.

| slice | class | N | PCD | PCD zeros-z | PCD scrambled-z | question-only | text name-free | text named | full file |
|---|---|---|---|---|---|---|---|---|---|
| in-distribution | categorical | 4880 | 0.844 | 0.630 | 0.780 | 0.862 | 0.843 | 0.857 | 0.873 |
| in-distribution | name_categorical | 2339 | 0.758 | 0.567 | 0.548 | 0.537 | 0.818 | 0.884 | 0.907 |
| in-distribution | name_point | 4039 | 0.812 | 0.745 | 0.682 | 0.705 | 0.846 | 0.914 | 0.929 |
| in-distribution | quant | 9608 | 0.793 | 0.638 | 0.739 | 0.800 | 0.802 | 0.807 | 0.831 |
| withheld names | name_point | 4508 | 0.813 | 0.743 | 0.681 | 0.697 | 0.869 | 0.926 | 0.936 |
| withheld race | name_categorical | 9311 | 0.713 | 0.582 | 0.507 | 0.570 | 0.789 | 0.832 | 0.844 |
| withheld race | name_point | 15243 | 0.741 | 0.709 | 0.626 | 0.703 | 0.824 | 0.904 | 0.915 |
| withheld phrasing | name_categorical | 4810 | 0.718 | 0.681 | 0.516 | 0.681 | 0.779 | 0.854 | 0.872 |
| withheld attribute | categorical | 12053 | 0.796 | 0.501 | 0.744 | 0.879 | 0.810 | 0.855 | 0.890 |

Notes. zeros-z isolates the decoder's question-plus-weights prior; scrambled-z substitutes another applicant's activation. question-only is the named-text auditor evaluated with an empty context, included as a second prior estimate. The withheld race is doubly out of distribution for PCD (encoder pretraining also excluded those applications) and singly for the text auditors.

## F. Primary comparison

PCD against the name-free text auditor on the withheld race with category questions, the most evenly matched cell. All 9,311 nonzero-delta questions, paired bootstrap (10,000 resamples), two-sided.

| quantity | value |
|---|---|
| PCD | 0.712 |
| text, name-free | 0.790 |
| difference | -0.078, 95% CI [-0.088, -0.068] |
| p (two-sided) | < 0.001 |
| label-sign ceiling for this cell | 0.919 |

The label ceiling reflects Monte-Carlo noise in the category-swap labels (mean over 16 sampled names); point-class labels are deterministic subject re-runs.

## G. Generation metrics (greedy decode, N=128 per class per slice)

sign accuracy / MAE / rate of predicting exactly zero. The PCD decoder's zero-rate is its hedging pathology; forced choice (table E) removes it.

| slice | class | PCD | text name-free | text named | full file |
|---|---|---|---|---|---|
| in-distribution | categorical | 0.586 / 31K / 0.41 | 0.766 / 17K / 0.21 | 0.805 / 20K / 0.16 | 0.828 / 13K / 0.16 |
| in-distribution | name_categorical | 0.750 / 32K / 0.12 | 0.695 / 34K / 0.02 | 0.961 / 18K / 0.00 | 0.961 / 14K / 0.00 |
| in-distribution | name_point | 0.430 / 51K / 0.55 | 0.656 / 32K / 0.24 | 0.797 / 21K / 0.14 | 0.797 / 16K / 0.16 |
| in-distribution | quant | 0.430 / 30K / 0.52 | 0.633 / 19K / 0.33 | 0.664 / 17K / 0.28 | 0.711 / 16K / 0.22 |
| withheld names | name_point | 0.375 / 45K / 0.59 | 0.586 / 32K / 0.34 | 0.828 / 13K / 0.15 | 0.859 / 10K / 0.13 |
| withheld race | name_categorical | 0.672 / 39K / 0.09 | 0.641 / 32K / 0.08 | 0.766 / 22K / 0.03 | 0.797 / 20K / 0.03 |
| withheld race | name_point | 0.422 / 50K / 0.56 | 0.703 / 35K / 0.25 | 0.898 / 19K / 0.09 | 0.914 / 13K / 0.09 |
| withheld phrasing | name_categorical | 0.672 / 35K / 0.06 | 0.688 / 29K / 0.08 | 0.789 / 18K / 0.02 | 0.812 / 15K / 0.04 |

Predict-zero baseline MAE is 45K on this sample. PCD sign accuracy when it commits to a nonzero answer is 0.91.

## H. Test-time k sweep (PCD, trained at k=16)

Generation on nonzero-delta questions, real z; teacher-forced CE on the mixed sample.

| test k | gen sign acc | committed sign acc | pred-zero rate | TF answer CE |
|---|---|---|---|---|
| 16 | 0.535 | 0.913 | 0.41 | 1.357 |
| 32 | 0.426 | 0.901 | 0.53 | 1.500 |
| 64 | 0.297 | 0.854 | 0.65 | 1.659 |

Widening the bottleneck at evaluation is out of distribution for the decoder and strictly hurts.

## I. Pretraining-scale variants

Identical fine-tuning recipe from the 9M and 18M encoder checkpoints; generation eval on the same nonzero-delta sample as table G.

| pretrain tokens | dictionary alive | val CE | sign acc (real z) | committed | name_categorical sign acc |
|---|---|---|---|---|---|
| 9M | 0.538 | 1.305 | 0.537 | 0.924 | 0.750 |
| 18M | 0.712 | 1.308 | 0.534 | 0.911 | 0.740 |
| 36M | 0.908 | 1.298 | 0.557 | 0.918 | 0.781 |

## J. Explanation study

gpt-5.4, temperature 0, no fine-tuning, no access to applications, names, or the formula. All outputs saved under `artifacts/explanations/llm_explanations/`; none discarded.

| tier | input | n | outcome |
|---|---|---|---|
| single application | one readout + one counterfactual question + the delta | 12 | 12/12 attribute the change to a name proxy, grounded in the readout |
| batch audit | 24 readouts + amounts, no questions | 8 | 8/8 flag gender-via-names as an impermissible factor |
| concept diff | top concept changes from one name swap | 12 | 12/12 identify name sensitivity; 4/12 recover the sign of the dollar effect |

## K. Runs

All training on 4xH100 (RunPod); reasoning regeneration on 1xH100. W&B project `pcd-counterfactuals` (user andrstic).

| run | W&B id |
|---|---|
| PCD pretraining (36M, v2) | mn7r5u9u |
| PCD decoder fine-tune | 30r32oqx |
| scale variants 9M / 18M | 439fotpm / 1rpiarmj |
| text auditor, named | ljukts7h |
| text auditor, name-free | tur80ah9 |
| full-file auditor | nax56hmp |

Weights are not committed. Encoder and LoRA checkpoints live under `artifacts/pcd/` and `artifacts/baselines/`; the concept readout parquet under `artifacts/pcd/pcd-finetune-v1/`; evaluation JSONs under `artifacts/baselines/evals/`.
