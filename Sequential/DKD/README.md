

# Decoupled Knowledge Distillation

[[official-code](https://github.com/megvii-research/mdistiller/blob/master/mdistiller/distillers/DKD.py)]


## Usage

Run with full-ranking

    python main.py --config=configs/xxx.yaml --ranking=full

or with sampled-based ranking

    python main.py --config=configs/xxx.yaml --ranking=pool