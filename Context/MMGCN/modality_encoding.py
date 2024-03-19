

# %%
import torch, os
import torch.nn as nn

from freeplot.utils import export_pickle
import freerec

# %%

root = "../../data"
dataset_ = "Amazon2023_Toys_10100_Chron"
device = torch.device("cuda:0")

mPath = "./Modality"
mFile = "feats.pkl"

text_converter = {
    'Title': lambda x: x,
    'Features': lambda x: ' '.join(eval(x)),
    'Description': lambda x: ' '.join(eval(x)),
    'Details': lambda x: ' '.join([f"{key}: {val}" for key, val in eval(x).items()])
}
text_encoder = "all-MiniLM-L6-v2"
cache_folder = "../../models"

# %%

dataset: freerec.data.datasets.RecDataSet = getattr(freerec.data.datasets.context, dataset_)(
    root
)
fields = dataset.fields

# %%


for field_name, converter in text_converter.items():
    field = fields[field_name]
    field.data = list(map(
        converter, field.data
    ))

texts = zip(*[fields[key].data for key in text_converter.keys()])
texts = list(map(lambda x: ' '.join(x), texts))
texts[:3]
# %%

from sentence_transformers import SentenceTransformer

model = SentenceTransformer(text_encoder, device=device, cache_folder=cache_folder)
tFeat = model.encode(texts)
tFeat[0]
# %%
