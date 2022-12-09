

# LightGCN


[[official-code](https://github.com/gusye1234/LightGCN-PyTorch)]


## Usage

`version: 0.1.1`

### Gowalla

    python main.py --config=Gowalla_m1.yaml


### AmazonCDs

```bash
python main.py --config=AmazonCDs_m1.yaml
```

| ReCall@20 | NDCG@20 |
| :-------: | :-----: |
|  0.1542   | 0.0971  |




### AmazonElectronics

```bash
python main.py --config=AmazonElectronics_m1.yaml
```

| ReCall@20 | NDCG@20 |
| :-------: | :-----: |
|  0.1235   | 0.0747  |



### AmazonBeauty

```bash
python main.py --config=AmazonBeauty_m1j.yaml
```

| ReCall@20 | NDCG@20 |
| :-------: | :-----: |
|  0.1689   | 0.0992  |
