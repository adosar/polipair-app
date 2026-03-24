import pandas as pd
from huggingface_hub import HfApi

df_pockets = pd.read_csv('Final_Receptor_dataset.csv')
df_pockets.to_parquet('hf://datasets/adosar/polipair-app-data/pocket_16k_features.parquet')

df_pubchem = pd.read_parquet('pubchem_ligands_cleared_5M_features.parquet')
df_pubchem.to_parquet('hf://datasets/adosar/polipair-app-data/pubchem_5M_features.parquet')
