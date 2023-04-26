

# BERT4Rec


## Usage


### MovieLens1M_550_Chron


    python main.py --config=configs/MovieLens1M_550_Chron.yaml


#### maxlen=100

| HITRATE@1 | HITRATE@5 | HITRATE@10 | HITRATE@20 | NDCG@5 | NDCG@10 | NDCG@20 |
| :-------: | :-------: | :--------: | :--------: | :----: | :-----: | :-----: |
|  0.4026   |  0.7091   |   0.8040   |   0.8788   | 0.5640 | 0.5945  | 0.6148  |


#### maxlen=200

| HITRATE@1 | HITRATE@5 | HITRATE@10 | HITRATE@20 | NDCG@5 | NDCG@10 | NDCG@20 |
| :-------: | :-------: | :--------: | :--------: | :----: | :-----: | :-----: |
|  0.4151   |  0.7228   |   0.8101   |   0.8877   | 0.5779 | 0.6058  | 0.6263  |


### AmazonBeauty_550_Chron


    python main.py --config=AmazonBeauty_550_Chron.yaml

| HITRATE@1 | HITRATE@5 | HITRATE@10 | HITRATE@20 | NDCG@5 | NDCG@10 | NDCG@20 |
| :-------: | :-------: | :--------: | :--------: | :----: | :-----: | :-----: |
|  0.1873   |  0.3773   |   0.4830   |   0.6055   | 0.2864 | 0.3204  | 0.3513  |


### AmazonGames_550_Chron

    python main.py --config=AmazonGames_550_Chron.yaml

| HITRATE@1 | HITRATE@5 | HITRATE@10 | HITRATE@20 | NDCG@5 | NDCG@10 | NDCG@20 |
| :-------: | :-------: | :--------: | :--------: | :----: | :-----: | :-----: |
|  0.3566   |  0.6379   |   0.7375   |   0.8262   | 0.5060 | 0.5381  | 0.5608  |