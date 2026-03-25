import numpy as np
import joblib
import streamlit as st
import pandas as pd
import polars as pl
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


def canonicalize_smiles(smiles: str):
    try:
        return Chem.CanonSmiles(smiles)
    except:
        return None


def smiles_to_desc(smiles: str):
    inp_smiles = canonicalize_smiles(smiles)

    if inp_smiles is None:
        return [None] * len(descriptor_names)
    else:
        mol = Chem.MolFromSmiles(inp_smiles)
        return list(descriptors.ComputeProperties(mol))


def ligands_to_desc(ligands: list[str]):
    df_desc = pd.DataFrame(
        [smiles_to_desc(l) for l in ligands],
        columns=descriptor_names
    )
    df_desc.insert(0, 'SMILES', ligands)

    return df_desc


@st.cache_data
def load_pockets():
    return pd.read_parquet(pockets_url).set_index('id')


@st.cache_resource
def load_model():
    return joblib.load('saved_models/best_model.joblib')


@st.cache_data
def filter_pubchem(amw_slider, nar_value, nha_value, fetch=False):
    ldf_filtered = ldf.filter(
        (pl.col('amw').is_between(amw_slider[0], amw_slider[1])) &
        (pl.col('NumAromaticRings') == nar_value) &
        (pl.col('NumHeavyAtoms') == nha_value)
    )
    if not fetch:
        return ldf_filtered.select(pl.len()).collect().item()
    else:
        return ldf_filtered.collect().to_pandas()



def prepare_inputs(X_poc, X_lig):
    X_input = pd.concat([X_poc] * len(X_lig))
    X_input.index = X_lig.index

    X_input = pd.concat([X_input, X_lig], axis=1)
    X_input.dropna(inplace=True)

    return X_input


def predict(X_poc, X_lig, from_pubchem):
    model = load_model()

    if len(X_lig) <= 20_000:  # Process directly if dataset is small

        X_input = prepare_inputs(X_poc, X_lig)
        smiles = X_input.pop('SMILES')

        preds = model.predict_proba(X_input)[:, 1]

        df_results = pd.DataFrame({
            'SMILES': smiles,
            'Score': preds
            })

        if from_pubchem:
            df_results.reset_index(inplace=True)

        return df_results

    else:
        return pd.concat([
            predict(X_poc, batch_lig, from_pubchem)
            for batch_lig in np.array_split(X_lig, n_chunks)
            ], axis=0)


descriptor_names = list(rdMolDescriptors.Properties.GetAvailableProperties())
descriptors = rdMolDescriptors.Properties(descriptor_names)
pockets_url = 'https://huggingface.co/datasets/adosar/polipair-app-data/resolve/main/pocket_16k_features.parquet'
pubchem_url = 'https://huggingface.co/datasets/adosar/polipair-app-data/resolve/main/pubchem_5M_features.parquet'
n_chunks = 5
ldf = pl.scan_parquet(pubchem_url)  # Lazy load PubChem
