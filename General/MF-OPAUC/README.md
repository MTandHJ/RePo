

## MF-OPAUC

[[official-code](https://github.com/swt-user/WWW_2023_code)]


**Note:** This implementation only includes `DNS` and `Softmax-based` losses with the uniform sampling. You may find that the calculation of importance is different from the official-code because it makes no sense when uniform sampling is applied.



## Usage


- DNS:

    python main.py --config=DNS/xxx.yaml

- Softmax:

    python main.py --config=Softmax/xxx.yaml