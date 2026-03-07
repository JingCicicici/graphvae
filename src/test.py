import pandas as pd
df = pd.read_parquet("/root/autodl-tmp/myproject/graphvae_repro_skeleton/runs/csi300/daily_metrics.parquet")
print(df.head())
print(df.columns)
