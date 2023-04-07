


# CAGCN


[[official-code](https://github.com/YuWVandy/CAGCN)]


## Usage

`version: 0.2.7`



### Gowalla_m1


#### jc

    python main.py --config=configs/Gowalla_m1.yaml --trend-type=jc --trend-coeff=1 -wd=1e-4 --fusion='FALSE'

    python main.py --config=configs/Gowalla_m1.yaml --trend-type=jc --trend-coeff=1.2 -wd=1e-3 --fusion='TRUE'

|                  | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 |
| :--------------: | :-------: | :-------: | :-----: | :-----: |
| CAGCN*-jc (best) |  0.1323   |  0.1883   | 0.1437  | 0.1595  |

#### lhn

    python main.py --config=configs/Gowalla_m1.yaml --trend-type=lhn --trend-coeff=1.2 -wd=1e-4 --fusion='FALSE'

    python main.py --config=configs/Gowalla_m1.yaml --trend-type=lhn --trend-coeff=1.2 -wd=1e-3 --fusion='TRUE'

#### sc

    python main.py --config=configs/Gowalla_m1.yaml --trend-type=sc --trend-coeff=1 -wd=1e-4 --fusion='FALSE'

    python main.py --config=configs/Gowalla_m1.yaml --trend-type=sc --trend-coeff=1.2 -wd=1e-3 --fusion='TRUE'


### Yelp2018_m1


#### jc

    python main.py --config=configs/Yelp2018_m1.yaml --trend-type=jc --trend-coeff=1.2 -wd=1e-4 --fusion='FALSE'

    python main.py --config=configs/Yelp2018_m1.yaml --trend-type=jc --trend-coeff=1.7 -wd=1e-3 --fusion='TRUE'

#### lhn

    python main.py --config=configs/Yelp2018_m1.yaml --trend-type=lhn --trend-coeff=1 -wd=1e-4 --fusion='FALSE'

    python main.py --config=configs/Yelp2018_m1.yaml --trend-type=lhn --trend-coeff=1 -wd=1e-3 --fusion='TRUE'

#### sc

    python main.py --config=configs/Yelp2018_m1.yaml --trend-type=sc --trend-coeff=1.2 -wd=1e-4 --fusion='FALSE'

    python main.py --config=configs/Yelp2018_m1.yaml --trend-type=sc --trend-coeff=1.7 -wd=1e-3 --fusion='TRUE'


### MovieLens1M_m2


#### jc

    python main.py --config=configs/MovieLens1M_m2.yaml --trend-type=jc --trend-coeff=2 -wd=1e-4 --fusion='FALSE'

    python main.py --config=configs/MovieLens1M_m2.yaml --trend-type=jc --trend-coeff=1 -wd=1e-3 --fusion='TRUE'

#### lhn

    python main.py --config=configs/MovieLens1M_m2.yaml --trend-type=lhn --trend-coeff=2 -wd=1e-4 --fusion='FALSE'

    python main.py --config=configs/MovieLens1M_m2.yaml --trend-type=lhn --trend-coeff=1 -wd=1e-3 --fusion='TRUE'

#### sc

    python main.py --config=configs/MovieLens1M_m2.yaml --trend-type=sc --trend-coeff=2 -wd=1e-4 --fusion='FALSE'

    python main.py --config=configs/MovieLens1M_m2.yaml --trend-type=sc --trend-coeff=1 -wd=1e-3 --fusion='TRUE'


### AmazonBooks_m1


#### jc

    python main.py --config=configs/AmazonBooks_m1.yaml --trend-type=jc --trend-coeff=2 -wd=1e-4 --fusion='FALSE'

    python main.py --config=configs/AmazonBooks_m1.yaml --trend-type=jc --trend-coeff=1.7 -wd=1e-3 --fusion='TRUE'

#### lhn

    python main.py --config=configs/AmazonBooks_m1.yaml --trend-type=lhn --trend-coeff=1 -wd=1e-4 --fusion='FALSE'

    python main.py --config=configs/AmazonBooks_m1.yaml --trend-type=lhn --trend-coeff=1.5 -wd=1e-3 --fusion='TRUE'

#### sc

    python main.py --config=configs/AmazonBooks_m1.yaml --trend-type=sc --trend-coeff=1 -wd=1e-4 --fusion='FALSE'

    python main.py --config=configs/AmazonBooks_m1.yaml --trend-type=sc --trend-coeff=1.7 -wd=1e-3 --fusion='TRUE'