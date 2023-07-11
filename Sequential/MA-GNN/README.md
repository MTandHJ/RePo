

# MA-GNN

[[official-code](https://github.com/Forrest-Stone/MA-GNN)]


**Note:** The official code adopts different left sequence lengths for training and testing, respectively. Here, they are the same.

## Usage

Run with full ranking:

    python main.py --config=configs/xxx.yaml --ranking=full

or with sampled-based ranking

    python main.py --config=configs/xxx.yaml --ranking=pool