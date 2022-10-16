

## LightGCN

> [He X., Deng K., Wang X., Li Y., Zhang Y. and Wang M. LightGCN: simplifying and powering graph convolution network for recommendation. In International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR), 2020.](http://arxiv.org/abs/2002.02126)


[[official-code](https://github.com/gusye1234/LightGCN-PyTorch)]


## Usage

**Note:** There are some unknown bugs existing in this implementation. When the loss decreases from `0.44`, it seems work well. But it will fall into a local optimal if from `0.68`. Sometimes meaningless codes like `print('???')` at the beginning can unexpectedly determine the result. I don't known this strange randomness is from `PyG` or `freerec` itself. So refer to [official-code](https://github.com/gusye1234/LightGCN-PyTorch) will be fun.


### Gowalla

    python main.py --config=Gowalla.yaml


