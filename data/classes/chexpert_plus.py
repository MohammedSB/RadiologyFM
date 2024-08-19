import os
import numpy as np
import pandas as pd

import torch
from PIL import Image

class CheXpertPlus(torch.utils.data.Dataset):
    def __init__(self, transform=None, path="/scratch/project_462000616/MFM/data/chexpert-plus"):
        self.df = pd.read_csv(os.path.join(path, "chexbert_labels" ,"df_chexpert_plus_240401.csv"))
        self.png_path = os.path.join(path, "PNG")
        self.transform = transform

        # get findings and impressions
        with open(os.path.join(path, "radgraph-XL-annotations", "section_findings.json")) as f:
            self.findings = json.load(f)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            item = self.df.iloc[idx]
            img_path = os.path.join(self.png_path, item["path_to_image"]).replace("jpg", "png")
            img = np.array(Image.open(img_path))
            if self.transform:
                img = self.transform(img)
            return img
        except:
            print("Error raised at idx", idx)