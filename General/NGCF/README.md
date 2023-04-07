

# NGCF

[[official-code](https://github.com/xiangwang1223/neural_graph_collaborative_filtering)]



## Usage

`version: 0.2.5`

### Gowalla

    python main.py --config=configs/Gowalla_m1.yaml

### Gowalla_10100811_Chron

    python main.py --config=configs/Gowalla_10100811_Chron.yaml

| Recall@10 | Recall@20 | NDCG@10 | NDCG@20 |
| :-------: | :-------: | :-----: | :-----: |
|  0.1112   |  0.1581   | 0.0955  | 0.1101  |


### Yelp2018_10104811_Chron

    python main.py --config=configs/Yelp2018_10104811_Chron.yaml

| Recall@10 | Recall@20 | NDCG@10 | NDCG@20 |
| :-------: | :-------: | :-----: | :-----: |
|  0.0376   |  0.0645   | 0.0246  | 0.0332  |


### MovieLens1M_10101811_Chron

    python main.py --config=configs/MovieLens1M_10101811_Chron.yaml

| Recall@10 | Recall@20 | NDCG@10 | NDCG@20 |
| :-------: | :-------: | :-----: | :-----: |
|  0.0575   |  0.1034   | 0.0789  | 0.0907  |


### AmazonCDs

    python main.py --config=configs/AmazonCDs_m1.yaml

| ReCall@20 | NDCG@20 |
| :-------: | :-----: |
|  0.1228   | 0.0746  |


### AmazonCDs_10104811_Chron

    python main.py --config=configs/AmazonCDs_10104811_Chron.yaml

| Recall@10 | Recall@20 | NDCG@10 | NDCG@20 |
| :-------: | :-------: | :-----: | :-----: |
|  0.0500   |  0.0830   | 0.0323  | 0.0426  |

### AmazonBooks_10104811_Chron

    python main.py --config=configs/AmazonBooks_10104811_Chron.yaml

| Recall@10 | Recall@20 | NDCG@10 | NDCG@20 |
| :-------: | :-------: | :-----: | :-----: |
|  0.0249   |  0.0402   | 0.0158  | 0.0206  |

### AmazonElectronics

    python main.py --config=configs/AmazonElectronics_m1.yaml

| ReCall@20 | NDCG@20 |
| :-------: | :-----: |
|  0.1177   | 0.0696  |

### AmazonElectronics_10104811_Chron


    python main.py --config=configs/AmazonElectronics_10104811_Chron.yaml

| Recall@10 | Recall@20 | NDCG@10 | NDCG@20 |
| :-------: | :-------: | :-----: | :-----: |
|  0.0298   |  0.0499   | 0.0184  | 0.0245  |


### AmazonBeauty

    python main.py --config=configs/AmazonBeauty_m1j.yaml

| ReCall@20 | NDCG@20 |
| :-------: | :-----: |
|  0.1466   | 0.0839  |
