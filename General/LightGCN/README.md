

# LightGCN


[[official-code](https://github.com/gusye1234/LightGCN-PyTorch)]


## Usage

`version: 0.2.5`

### Gowalla_m1

    python main.py --config=configs/Gowalla_m1.yaml

### Gowalla_10100811_Chron

    python main.py --config=configs/Gowalla_10100811_Chron.yaml

| Recall@10 | Recall@20 | NDCG@10 | NDCG@20 |
| :-------: | :-------: | :-----: | :-----: |
|  0.1276   |  0.1784   | 0.1105  | 0.1262  |


### Yelp2018_10104811_Chron

    python main.py --config=configs/Yelp2018_10104811_Chron.yaml

| Recall@10 | Recall@20 | NDCG@10 | NDCG@20 |
| :-------: | :-------: | :-----: | :-----: |
|  0.0440   |  0.0733   | 0.0283  | 0.0378  |


### MovieLens1M_10101811_Chron

    python main.py --config=configs/MovieLens1M_10101811_Chron.yaml

| Recall@10 | Recall@20 | NDCG@10 | NDCG@20 |
| :-------: | :-------: | :-----: | :-----: |
|  0.0617   |  0.1075   | 0.0824  | 0.0938  |

### AmazonCDs

```bash
python main.py --config=configs/AmazonCDs_m1.yaml
```

| Recall@20 | NDCG@20 |
| :-------: | :-----: |
|  0.1542   | 0.0971  |

### AmazonCDs_10104811_Chron

    python main.py --config=configs/AmazonCDs_10104811_Chron.yaml

| Recall@10 | Recall@20 | NDCG@10 | NDCG@20 |
| :-------: | :-------: | :-----: | :-----: |
|  0.0600   |  0.0959   | 0.0387  | 0.0500  |


### AmazonBooks_10104811_Chron

    python main.py --config=configs/AmazonBooks_10104811_Chron.yaml

| Recall@10 | Recall@20 | NDCG@10 | NDCG@20 |
| :-------: | :-------: | :-----: | :-----: |
|  0.0314   |  0.0526   | 0.0197  | 0.0264  |

### AmazonElectronics

    python main.py --config=configs/AmazonElectronics_m1.yaml

| Recall@20 | NDCG@20 |
| :-------: | :-----: |
|  0.1235   | 0.0747  |


### AmazonElectronics_10104811_Chron


    python main.py --config=configs/AmazonElectronics_10104811_Chron.yaml

| Recall@10 | Recall@20 | NDCG@10 | NDCG@20 |
| :-------: | :-------: | :-----: | :-----: |
|  0.0340   |  0.0538   | 0.0211  | 0.0271  |

### AmazonBeauty

    python main.py --config=configs/AmazonBeauty_m1.yaml

| Recall@20 | NDCG@20 |
| :-------: | :-----: |
|  0.1689   | 0.0992  |
