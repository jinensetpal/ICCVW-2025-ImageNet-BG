import os
import numpy as np
import pandas as pd
import json

json_file_path = 'imagenet_class_index.json'
with open(json_file_path, 'r') as file:
    imagenet_class_index = json.load(file)

dfs = []

model_order = [
    'ResNet50_7x7_1',
    'ResNet50_GALS_7x7_1',
    'ResNet50_GALS_7x7_2',
    'ResNet50_GALS_7x7_3',
    'ResNet50_GALS_14x14_1',
    'ResNet50_GALS_14x14_2',
    'ResNet50_GALS_14x14_3',
    'ResNet50_ABN_14x14_1',
    'ResNet50_ABN_14x14_2',
    'ResNet50_ABN_14x14_3',
    'ResNet50_14x14_1',
    'ResNet50_14x14_2',
    'ResNet50_14x14_3',
    'ResNet50_CFCE_7x7_1',
    'ResNet50_CFCE_7x7_2',
    'ResNet50_CFCE_7x7_3',
    'ResNet50_CFCE_14x14_1',
    'ResNet50_CFCE_14x14_2',
    'ResNet50_CFCE_14x14_3',
]


files_path = "./model_results/"
files = [os.path.join(files_path, f) for f in os.listdir(files_path) if f.endswith(".pickle")]
for file in files:
    model_name = file.split("/")[-1].split(".")[0]
    if model_name not in model_order: continue
    print(f"> {model_name}")
    
    df = pd.read_pickle(file)
    df["prediction"] = df["logits"].apply(np.argmax)
    df["model"] = model_name
    dfs.append(df)
    
df = pd.concat(dfs, ignore_index=True)
df = df[df['quality'].isin(['good', 'v_good'])]

for grp, subset in df.groupby(['type', 'model']):
    print(grp, f"{(subset['label'] == subset['prediction']).mean():.7f}")
