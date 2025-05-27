import pandas as pd
import os
from scipy.stats import ttest_1samp

folder = 'draw_dataset'
files = [f for f in os.listdir(folder) if f.endswith('.csv')]

results = []

for fname in files:
    path = os.path.join(folder, fname)
    df = pd.read_csv(path)

    # 自动适配列名
    for col, cname in [('improvement', '公平提升'), ('IFCM - FCM', '轮廓系数提升')]:
        vals = df[col].dropna().values
        t_stat, p_val = ttest_1samp(vals, 0.0)
        print(f"{fname[:-4]}: {cname} t={t_stat:.4f}, p={p_val:.4f}, mean={vals.mean():.4f}, n={len(vals)}")
        results.append(
            {'dataset': fname[:-4], 'metric': cname, 't': t_stat, 'p': p_val, 'mean': vals.mean(), 'n': len(vals)})

pd.DataFrame(results).to_excel('t_test_results.xlsx', index=False)
