# eks-rag-acceptance 对拍报告

查询数 1500；切片 enterprise, standard, supplement

### dense

| 系统 | hit@5 | recall@5 | mrr@10 | ndcg@10 | reject@5 | abstain@5 | p50(ms) | p95(ms) | QPS |
|---|---|---|---|---|---|---|---|---|---|
| rust | 0.8971 | 0.7485 | 0.894 | 0.7827 | 0.29 | 0.69 | 30.3 | 40.0 | 31.784 |

### sparse

| 系统 | hit@5 | recall@5 | mrr@10 | ndcg@10 | reject@5 | abstain@5 | p50(ms) | p95(ms) | QPS |
|---|---|---|---|---|---|---|---|---|---|
| rust | 0.9386 | 0.8192 | 0.8976 | 0.8179 | 0.04 | 0.96 | 4.2 | 6.0 | 234.412 |

### hybrid

| 系统 | hit@5 | recall@5 | mrr@10 | ndcg@10 | reject@5 | abstain@5 | p50(ms) | p95(ms) | QPS |
|---|---|---|---|---|---|---|---|---|---|
| rust | 0.9993 | 0.9515 | 0.981 | 0.9424 | 0.02 | 0.96 | 68.4 | 1357.2 | 2.711 |
