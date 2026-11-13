# eks-rag-acceptance 对拍报告

查询数 1500；切片 enterprise, standard, supplement

### dense

| 系统 | hit@5 | recall@5 | mrr@10 | ndcg@10 | reject@5 | abstain@5 | p50(ms) | p95(ms) | QPS |
|---|---|---|---|---|---|---|---|---|---|
| rust | 0.8971 | 0.7485 | 0.894 | 0.7827 | 0.29 | 0.69 | 37.4 | 52.8 | 26.312 |

### sparse

| 系统 | hit@5 | recall@5 | mrr@10 | ndcg@10 | reject@5 | abstain@5 | p50(ms) | p95(ms) | QPS |
|---|---|---|---|---|---|---|---|---|---|
| rust | 0.9386 | 0.8192 | 0.8976 | 0.8179 | 0.04 | 0.96 | 5.8 | 10.4 | 159.507 |

### hybrid

| 系统 | hit@5 | recall@5 | mrr@10 | ndcg@10 | reject@5 | abstain@5 | p50(ms) | p95(ms) | QPS |
|---|---|---|---|---|---|---|---|---|---|
| rust | 0.9993 | 0.9514 | 0.9781 | 0.9389 | 0.03 | 0.95 | 88.5 | 270.0 | 8.511 |
