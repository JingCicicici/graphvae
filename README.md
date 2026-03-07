# GraphVAE (CIKM'24) reproduction skeleton (PyTorch + Qlib)

This is a starter repo to reproduce the paper:
**GraphVAE: Unveiling Dynamic Stock Relationships with Variational Autoencoder-based Factor Modeling**.

It implements the core equations in the paper (dynamic graph from cosine similarity + temporal aggregation, GRU feature extractor, graph relation update, VAE-style posterior/prior + decoder, and the loss).

> Notes
- This repo is a *skeleton*: it aims to be correct-by-construction for the paper's described math, but you still need to plug in data and tune hyperparameters to match the paper's reported numbers.
- The paper uses Qlib Alpha158 on CSI300/CSI500/CSI1000 with rolling training (5y train, 1y valid, 1y test).

## 1) Environment

```bash
conda create -n graphvae python=3.10 -y
conda activate graphvae

pip install -r requirements.txt
```

## 2) Download Qlib China A-share data (bin)

If you use Qlib's official dataset:

```bash
git clone https://github.com/microsoft/qlib
cd qlib
pip install -e .
python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn --interval 1d
```

## 3) Quick smoke test (random data)

```bash
python -m src.train --smoke_test 1
```

## 4) Real training with dumped Alpha158 panel

1) Dump Alpha158 features/labels into npz files:

```bash
python -m src.data.dump_qlib_alpha158 --provider_uri ~/.qlib/qlib_data/cn_data --market csi300 --out_dir ./data/csi300
```

2) Train:

```bash
python -m src.train --data_dir ./data/csi300 --market csi300
```

3) Evaluate TopK-Drop:

```bash
python -m src.eval --pred_path ./runs/csi300/predictions.parquet
```
