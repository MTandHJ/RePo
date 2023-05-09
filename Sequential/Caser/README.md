

# Caser


[[official-code](https://github.com/graytowne/caser_pytorch)]


**Note:** Following the official code, we train Caser with all possible sequences generated in a rolling fashion.
As a result, Caser actually sees more interactions than other models such as SASRec.
However, Caser uses a limited amount of historical information (e.g., `maxlen=5`) to predict the next item.

## Usage


Run with sampled-based ranking:

    python main.py --config=configs/xxx.yaml

or with full-ranking

    python main_full_ranking.py --config=configs/xxx.yaml

