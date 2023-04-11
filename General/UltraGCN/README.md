

# UltraGCN


[[official-code](https://github.com/xue-pai/UltraGCN)]



## Usage

The hyper-parameters of `neg_weight`, `item_weight` and `norm_weight` are re-searched:

    neg_weight: [100, 200, 300, 400, 500]
    item_weight: [1.e-8, 1.e-6, 1.e-4, 1.e-2, 0., 1., 2., 3., 4., 5.]
    norm_weight: [0., 1.e-5, 1.e-4, 1.e-3, 1.e-2]

### UltraGCN_base

    python main.py --config=base_configs/xxx.yaml


### UltraGCN

    python main.py --config=configs/xxx.yaml