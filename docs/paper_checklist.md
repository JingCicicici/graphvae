# GraphVAE paper checklist

Paper: GraphVAE: Unveiling Dynamic Stock Relationships with Variational Autoencoder-based Factor Modeling (CIKM 2024)

## 1. Task definition
- Predict cross-sectional next-day stock return ranking scores.
- Input at time s: X_s in R^{T x N x C}
  - T: past time steps
  - N: number of stocks at date s
  - C: number of characteristics
- Label:
  y_s^i = (price_{s+1}^i - price_s^i) / price_s^i

## 2. Dynamic relationship graph
For each day in the past T days:
1. For each stock pair (i, j), compute cosine similarity using same-day feature vectors:
   S_ij^t = cos(x_i^t, x_j^t)
2. Use threshold tau as the eta-quantile of similarity matrix S^t
3. Keep only similarities above tau:
   R_t(i, j) = S_ij^t if S_ij^t > tau else 0

Temporal aggregation:
- Weight each day by exponential decay:
  w_t = alpha^(T - t)
- Aggregate:
  R_a(i, j) = sum_t w_t * R_t(i, j)

Final graph:
- For each stock i, keep top-k most similar stocks j
- Build a binary graph G = (V, E)

## 3. Stock feature extractor
- Use GRU on historical stock data X
- Output latest hidden state e_s in R^{N x H}

## 4. Graph-based feature refinement
Compute relation strength:
- eta_{i,j} = attention-like normalized score over neighbors

Update representation:
- e_hat_i = e_i + sum_{j in N_i} eta_{i,j} * e_j

## 5. Posterior encoder
- Posterior uses refined stock features e_hat_s and future returns y
- Output mu_post, sigma_post
- Sample z ~ N(mu_post, diag(sigma_post))

Specific operation:
- First project e_hat_s with linear layer
- Then element-wise multiply with future return y
- Then map to posterior mean and std

## 6. Prior predictor
- Prior uses only refined stock features e_hat_s
- Output mu_prior, sigma_prior
- Sample z_prior ~ N(mu_prior, diag(sigma_prior^2))

## 7. Decoder
- Decoder predicts returns with:
  y_hat = alpha + beta * z
- alpha is idiosyncratic return branch
- beta is factor exposure branch from e_hat_s

## 8. Loss
Total loss has two parts:
1. reconstruction term on future returns
2. KL divergence between posterior and prior

## 9. Dataset and evaluation
- Chinese market: CSI300 / CSI500 / CSI1000
- Daily data from 2013-01-01 to 2023-12-31
- Features: Qlib Alpha158
- Rolling training:
  - train 5 years
  - valid 1 year
  - test 1 year
- CSI1000 uses shorter rolling setup because components are published later

## 10. Portfolio evaluation
- TopK-Drop strategy
- K = 50
- Drop N = 5
- Include transaction costs:
  - buy 0.05%
  - sell 0.15%
- Metrics:
  - AR
  - IR
  - IC
  - RankIC

## 11. Hyperparameters explicitly mentioned in paper
- eta = 90
- alpha = 0.9
- k = 5

## 12. Important ambiguity to verify manually
- The paper is short and does not fully specify all implementation details.
- Need to manually verify:
  - exact GRU hidden size
  - latent factor dimension K
  - optimizer / lr / batch size
  - exact decoder implementation details
  - exact rolling split code
  - exact stock universe filtering and missing-data handling

## 13. Code audit targets
Please check whether the repository implementation matches the paper on:
1. label definition
2. input tensor shape and slicing
3. dynamic graph construction
4. temporal aggregation weights
5. top-k graph building
6. GRU feature extractor
7. posterior encoder uses future returns
8. prior predictor does not use future returns
9. decoder formula and branches
10. loss = reconstruction + KL
11. rolling split and evaluation protocol
12. TopK-Drop backtest settings
