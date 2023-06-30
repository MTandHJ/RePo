

## MF-OPAUC

[[official-code](https://github.com/swt-user/WWW_2023_code)]

**Note:** This implementation only includes `DNS` and `Softmax-based` losses with the uniform sampling. You may find that the calculation of importance is different from the official-code because it makes no sense when uniform sampling is applied.


## Usage

Run with full-ranking

    python main.py --config=configs/xxx.yaml --ranking=full

or with sampled-based ranking

    python main.py --config=configs/xxx.yaml --ranking=pool