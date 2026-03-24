from tqdm import tqdm
import pandas as pd
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


def ligands_to_desc(ligands: list[str], cids: list[str]):
    df = pd.DataFrame(
        [smiles_to_desc(l) for l in tqdm(ligands, desc='Featurizing PubChem')],
        columns=descriptor_names
    )
    df['SMILES'] = ligands
    df['CID'] = cids

    return df


descriptor_names = list(rdMolDescriptors.Properties.GetAvailableProperties())
descriptors = rdMolDescriptors.Properties(descriptor_names)

data = pd.read_parquet('pubchem_ligands_cleared_5M.parquet.gz')
df = ligands_to_desc(ligands=list(data.smiles_canonical), cids=list(data.CID))
df.to_parquet('pubchem_ligands_cleared_5M_features.parquet')
