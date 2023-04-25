

# STOSA


[official-code](https://github.com/zfan20/STOSA)



**Note:** The `Beauty` dataset used in the original paper is different from ours. We also test our code on the `Beauty` dataset given by the author. The offical code gives (on test set, `seed=0`) `NDCG@5=0.0321` and `NDCG@10=0.0385`, while our code gives `NDCG@5=0.0307` and `NDCG@10=0.0372`; however, both are inferior to the results reported in the paper.

## Usage


Run with sampled-based ranking:

    python main.py --config=configs/xxx.yaml

or with full-ranking

    python main_full_ranking.py --config=configs/xxx.yaml

