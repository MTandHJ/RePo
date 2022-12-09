

# NGCF

[[official-code](https://github.com/xiangwang1223/neural_graph_collaborative_filtering)]



## Usage

`version: 0.1.1`

### Gowalla

```bash
python main.py --config=Gowalla_m1.yaml
```


### AmazonCDs

```bash
python main.py --config=AmazonCDs_m1.yaml
```

| ReCall@20 | NDCG@20 |
| :-------: | :-----: |
|  0.1228   | 0.0746  |




### AmazonElectronics

```bash
python main.py --config=AmazonElectronics_m1.yaml
```

| ReCall@20 | NDCG@20 |
| :-------: | :-----: |
|  0.1177   | 0.0696  |



### AmazonBeauty

```bash
python main.py --config=AmazonBeauty_m1j.yaml
```

| ReCall@20 | NDCG@20 |
| :-------: | :-----: |
|  0.1466   | 0.0839  |
