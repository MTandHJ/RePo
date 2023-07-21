

# BERT4Rec-emb

**Note:** BERT4Rec-emb uses embeddings for final classification instead of a full connected layer.

## Usage

Run with full ranking:

    python main.py --config=configs/xxx.yaml --ranking=full

or with sampled-based ranking

    python main.py --config=configs/xxx.yaml --ranking=pool