# My current questions for Codex

This repository is my reproduction / adaptation attempt of the GraphVAE paper.

I do not want a blind rewrite.
I want a code audit first:
- find where the implementation matches the paper
- find where it does not match
- identify possible bugs
- give minimal fixes with clear reasoning

## 1. Main goal

Please check whether my current code is consistent with the GraphVAE paper, especially in the following aspects:

1. data pipeline
2. label construction
3. rolling split
4. dynamic graph construction
5. model architecture
6. posterior / prior design
7. decoder formula
8. loss function
9. evaluation metrics
10. backtest settings

Please do not immediately refactor everything.
First produce an audit report.

---

## 2. My current concerns

### 2.1 Dataset coverage may be wrong
I am worried that the actual data used by the current pipeline does not fully cover the paper setting.

Possible issues:
- the paper uses data from 2013-01-01 to 2023-12-31
- my current run may only cover data until around 2020
- I may have generated multiple `.npz` files with inconsistent date ranges
- the training / validation / testing windows may not actually use the latest data file

Please check:
- which data file is really being loaded
- the exact date range of the loaded panel
- whether the code truly uses 2013-01-01 to 2023-12-31
- whether there are stale files or old paths still being used

---

### 2.2 Rolling split may not match the paper
I am worried that my rolling split implementation is not exactly the same as the paper.

Please verify:
- whether the code uses 5 years for training, 1 year for validation, and 1 year for testing
- whether the rolling window advances correctly
- whether split boundaries are based on calendar years or index positions
- whether the CSI1000 special case is handled separately if needed
- whether there is any leakage across train / valid / test

Also tell me:
- where the split is implemented
- whether the code logic matches the intended paper protocol

---

### 2.3 Label definition may be mismatched
I am not fully sure whether my label is constructed exactly as the paper expects.

Please check:
- whether label is next-day return
- whether the shift direction is correct
- whether the label uses raw return or log return
- whether the denominator and timing are correct
- whether any future information leaks into features

I want a very explicit answer:
- what is the current label formula in code
- what should it be according to the paper
- whether the current implementation is correct

---

### 2.4 Input tensor shape may be inconsistent
I am worried that the model input shape may not follow the intended `T x N x C` logic.

Please check:
- what shape the dataset returns
- whether `T`, `N`, and `C` are in the expected order
- whether the GRU input matches the dataset output
- whether batching changes the expected shape
- whether there are silent shape mismatches that do not crash but still produce wrong learning behavior

Please clearly list:
- dataset output shape
- model input shape
- intermediate shapes in the encoder
- whether they are consistent

---

### 2.5 Dynamic graph construction may not match the paper
This is one of my biggest concerns.

I am not sure whether my graph building logic really follows the paper.

Please verify step by step:

1. do we compute pairwise cosine similarity for each day separately?
2. do we threshold each daily similarity matrix by quantile / percentile?
3. do we keep only similarities above the threshold?
4. do we apply exponential temporal decay across the past `T` days?
5. do we aggregate the daily graphs into one final relationship matrix?
6. do we keep top-k neighbors for each stock?
7. is the final graph binary or weighted?
8. is graph construction done using the same stock universe alignment as the features?

Please also point out:
- whether my code uses correlation instead of cosine similarity
- whether it uses the whole window at once instead of day-by-day graph construction
- whether `top-k`, thresholding, and temporal weighting are implemented correctly

---

### 2.6 Hyperparameters may be inconsistent
I am not sure whether key hyperparameters follow the paper.

Please inspect:
- `eta` / quantile threshold
- `alpha` / temporal decay factor
- `k` / top-k neighbors
- lookback window length `T`
- GRU hidden size
- latent factor dimension
- learning rate
- batch size
- optimizer
- loss weights

Please tell me:
- which values are explicitly set in code
- which values are missing
- which values are inconsistent with the paper
- which values are currently hard-coded in unclear places

---

### 2.7 Posterior / prior implementation may have information leakage
This is another high-priority issue.

Please check very carefully:

- does the posterior encoder use future returns?
- does the prior predictor avoid using future returns?
- during evaluation / test time, is only the prior used?
- is there any accidental leakage of labels into inference?

I want you to explicitly identify:
- where posterior is implemented
- where prior is implemented
- what inputs each one receives
- whether the current code is safe

If there is any data leakage risk, mark it as high priority.

---

### 2.8 Decoder may not match the factor-model form
I am not sure whether the decoder really follows the paper's factor model style.

Please verify:
- whether the decoder predicts `y_hat = alpha + beta * z`
- whether `alpha` is implemented as an idiosyncratic branch
- whether `beta` is produced from stock features
- whether the latent factor `z` is used correctly
- whether dimensions are consistent

If the current implementation is only a generic MLP head or a simplified regression head, please point that out clearly.

---

### 2.9 Loss function may be incomplete
Please check whether the current loss is really the intended VAE-style objective.

Specifically verify:
- reconstruction loss exists
- KL divergence exists
- the total loss is composed correctly
- KL weight is reasonable
- the code actually uses the intended sampled latent variable
- training does not accidentally bypass part of the VAE design

Please clearly state:
- what the current loss is
- whether it matches the paper
- whether any part is missing or ineffective

---

### 2.10 Evaluation and backtest may not match the paper
I am worried that even if the model runs, the evaluation protocol may not match the paper.

Please verify:
- whether IC and RankIC are computed correctly
- whether AR and IR are computed correctly
- whether the backtest uses TopK-Drop
- whether `K=50` and `Drop=5` are used
- whether transaction costs are included
- whether buy / sell costs match the paper
- whether benchmark / portfolio logic is consistent

Also check:
- whether evaluation uses prediction scores from the correct dates
- whether prediction and realized return alignment is correct
- whether any date mismatch exists in backtest

---

### 2.11 Current repository may contain old experimental code
I suspect my repository may contain old generated code, partially outdated files, or paths that are no longer actually used.

Please help identify:
- unused files
- duplicate data loaders
- duplicate graph-building logic
- stale scripts that are no longer in the main training pipeline
- files that appear to be AI-generated placeholders but are not truly connected to the workflow

I want to know:
- what files are actually on the main execution path
- what files are dead code
- what files are likely causing confusion

---

## 3. What I want from the audit output

Please produce the audit in this format:

### A. Repository execution path
- which files are actually used for training
- which files are actually used for inference / evaluation / backtest

### B. Paper-to-code mapping
For each key paper component, tell me:
- corresponding file
- corresponding class / function
- whether it matches

### C. Mismatch list
For each mismatch:
- severity: high / medium / low
- what the paper expects
- what the code currently does
- why it is a problem

### D. Minimal fix plan
Do not rewrite the entire project.
Instead provide:
- the smallest set of files that should be changed first
- the order of fixes
- a short validation method for each fix

### E. High-risk bug list
Please explicitly flag:
- data leakage
- wrong date alignment
- wrong return label
- incorrect rolling split
- graph construction inconsistent with the paper
- evaluation misalignment
- backtest setting mismatch

---

## 4. Constraints for modification

If you later modify code, please follow these rules:

- do not arbitrarily change the whole repository structure
- do not rename many files unless necessary
- do not touch unrelated code
- prefer minimal, local fixes
- explain every changed file
- after each change, explain how to verify it

---

## 5. Extra note

This repository may contain multiple versions of dataset files and scripts.
Please do not assume the newest-looking file is the one actually used.
Trace the real execution path from the training entry script.

My priority is not "make it run at any cost".
My priority is:
1. make the implementation logically consistent with the paper
2. avoid leakage and alignment bugs
3. ensure evaluation is trustworthy
