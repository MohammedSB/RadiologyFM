#%%
import os
import torch
import pandas as pd
import json
from tqdm import tqdm

#%%
print(torch.cuda.is_available())
df = pd.read_csv("/scratch/project_462000616/MFM/data/chexpert-plus/chexbert_labels/df_chexpert_plus_240401.csv")
#%%
import os
missing = []
path = "/scratch/project_462000616/MFM/data/chexpert-plus/PNG" 
for d in tqdm(df.iterrows()):
    img_path = os.path.join(path, d[1]["path_to_image"]).replace("jpg", "png")
    if not os.path.isfile(img_path):
        missing.append(img_path) 

#%%
len(missing)
# %%
