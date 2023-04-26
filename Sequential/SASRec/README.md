


# SASRec


## Usage


### MovieLens1M_550_Chron


    python main.py --config=configs/MovieLens1M_550_Chron.yaml


#### maxlen=50

| HITRATE@1 | HITRATE@5 | HITRATE@10 | NDCG@5 | NDCG@10 |
| :-------: | :-------: | :--------: | :----: | :-----: |
|  0.3368   |  0.6785   |   0.7934   | 0.5162 | 0.5530  |


#### maxlen=200

| HITRATE@1 | HITRATE@5 | HITRATE@10 | NDCG@5 | NDCG@10 |
| :-------: | :-------: | :--------: | :----: | :-----: |
|  0.3737   |  0.7159   |   0.8194   | 0.5548 | 0.5877  |



### AmazonBeauty_550_Chron


    python main.py --config=AmazonBeauty_550_Chron.yaml

| HITRATE@1 | HITRATE@5 | HITRATE@10 | NDCG@5 | NDCG@10 |
| :-------: | :-------: | :--------: | :----: | :-----: |
|  0.2409   |  0.4233   |   0.5181   | 0.3363 | 0.3668  |


### AmazonGames_550_Chron

    python main.py --config=AmazonGames_550_Chron.yaml

| HITRATE@1 | HITRATE@5 | HITRATE@10 | NDCG@5 | NDCG@10 |
| :-------: | :-------: | :--------: | :----: | :-----: |
|  0.3504   |  0.6390   |   0.7373   | 0.5042 | 0.5359  |
