import pandas as pd
import streamlit as st

from utils import ligands_to_desc, load_pockets, predict, filter_pubchem

logo_link = r'https://raw.githubusercontent.com/adosar/polipair-app/master/images/logo.png'

# Configuration
# =======================================
st.set_page_config(
        page_title='PoLiPaiR',
        #page_icon='🤖',
        layout='wide',
        initial_sidebar_state='expanded',
        menu_items={
            'Get Help': 'https://github.com/adosar/polipair/discussions',
            'Report a bug': 'https://github.com/adosar/polipair/issues',
            }
        )
check_icon = ':material/check_circle:'
info_icon = ':material/info:'
warn_icon = ':material/warning:'
error_icon = ':material/error:'
pubchem_lim = 100_000
preds_show_lim = 50_000


# Sidebar
# =======================================
st.sidebar.header('About')
st.sidebar.markdown(f'PoLiPaiR demo built with **:red[Streamlit]**.', unsafe_allow_html=True)
st.sidebar.link_button(
        ':material/deployed_code_account: Created by Antonios P. Sarikas',
        'https://github.com/adosar',
        )
st.sidebar.subheader('Disclaimer')
st.sidebar.caption(f'''
This app uses AI models to make predictions. Please note that these predictions
are intended for informational purposes only. By using this app, you agree that
any decisions made based on the predictions are at your own risk. The creator of
this app is not responsible for any outcomes resulting from your use of the
predictions provided. For any high-risk or critical decisions, consider
consulting with a qualified professional or using additional resources to verify
the predictions.

Uploaded files are stored in memory and get deleted immediately as soon as
they’re not needed anymore.  For more information about data file handling,
please refer to the official [Streamlit
documentation](https://docs.streamlit.io/knowledge-base/using-streamlit/where-file-uploader-store-when-deleted).
Use at your own risk.
''', unsafe_allow_html=True)

st.logo(logo_link, size='large')
st.sidebar.subheader('How to Cite')
st.sidebar.write('If you find PoLiPaiR useful, please consider citing us.', unsafe_allow_html=True)

bibtex = 'Currently N/A.'
citation_text = 'Currently N/A.'

cols = st.sidebar.columns(2)
with cols[0]:
    with st.popover('BibTeX', icon=':material/article:'):
        st.code(bibtex, language=None)

with cols[1]:
    with st.popover('Other', icon=':material/article:'):
        st.code(citation_text, language=None)

# Titlebar
# =======================================
st.markdown(rf"""
<h1 align="center">
  <img alt="Logo"
  src="{logo_link}" width=50%/>
</h1>
""", unsafe_allow_html=True)
st.divider()
st.title('🎉 Welcome to PoLiPaiR')
st.markdown("""
**PoLiPaiR** is a machine learning model designed to **evaluate the
fitness between a protein pocket and a candidate ligand**. By leveraging
biochemical and physicochemical features extracted from the pocket and the
ligand, respectively, PoLiPaiR predicts how well a given pair is likely to
match. Under the hood, a trained ML model transforms these features into a
quantitative compatibility score, enabling fast ranking of pocket-ligand pairs.

PoLiPaiR supports three core workflows:

1. **Score a single pocket–ligand pair**.
2. **Rank a list of candidate pockets for a given target ligand**.
3. **Rank a list of candidate ligands for a given target pocket**.

At the moment, the demo provides only the second option — you can select a
target protein pocket from a library of over 16,000 pockets, select your
candidate ligands, and receive a ranked list based on their predicted fitness
scores.
""")


# Pocket selection
# =======================================
st.header('🔬 Select pocket to analyze', divider=True)
st.markdown('Select the pocket to analyze by typing its `PDB ID` below:')

pdb_id = st.text_input(
    'Enter `PDB ID`',
    placeholder='Example of valid entry: 10gs',
    value=None,
)

X_pocket = None
df_pockets = load_pockets()

if pdb_id is not None:
    try:
        X_pocket = df_pockets.loc[[pdb_id.lower()]]
        with st.expander('Show features of selected pocket'):
            st.write(X_pocket.iloc[0])
    except:
        st.error(f'PDB ID "{pdb_id}" not found in the database. Please try again.', icon=error_icon)


# Ligands upload
# =======================================
st.header('📋 Select candidate ligands', divider=True)

option_list = [
    'I will upload my own ligands',
    'I will use fragment-based ligands from PubChem',
    ]

option = st.selectbox(
    'How would you like to provide candidate ligands?',
    option_list,
    index=None,
    placeholder='Select an option',
)

X_ligands = None

if option == option_list[0]:
    st.markdown("""
    The file **must include a column named `smiles`** containing the SMILES
    representation of each ligand.
    """)

    with st.expander('Example of a valid `.csv` file'):
        st.markdown("""
        ```csv
        smiles
        CC
        CCC
        CCO
        ```
        """)

    uploaded_file = st.file_uploader(
            'Upload a `.csv` file containing SMILES of ligands',
            type='csv',
            max_upload_size=1
            )
    if uploaded_file:
        df_ligands = pd.read_csv(uploaded_file)
        with st.expander('Show features of uploaded ligands'):
            X_ligands = ligands_to_desc(list(df_ligands.smiles))
            from_pubchem = False
            st.dataframe(X_ligands, hide_index=True)

elif option == option_list[1]:
    st.info('''You can **choose up to 100,000** ligands from a total of 5
            million sourced from PubChem. Use the sliders to narrow down the
            list of candidate ligands.''', icon=info_icon)


    amw_slider = st.slider(
            'molecular weight',
            min_value=50.,
            max_value=300.,
            value=(150., 200.),
            step=0.1,
            )

    col1, col2 = st.columns(2)
    with col1:
        nar_value = st.slider(
                'number of aromatic rings',
                min_value=0,
                max_value=9,
                value=1,
                step=1,
                )
    with col2:
        nha_value = st.slider(
                'number of heavy atoms',
                min_value=0,
                max_value=34,
                value=8,
                step=1,
                help='The total count of non-hydrogen atoms',
                )

    count = filter_pubchem(amw_slider, nar_value, nha_value)

    if count > pubchem_lim:
        st.warning(
            f'Budget exceeded: {count:,} ligands. Please refine your filters.',
            icon=warn_icon
        )
    elif count == 0:
        st.warning(
            'No ligands found. Please refine your filters.',
            icon=warn_icon
        )
    else:
        agree = st.checkbox(f'**{count:,}** ligands can be fetched from PubChem. Do you want to continue?')
        if agree:
            with st.spinner(f'Fetching **{count:,}** ligands from PubChem. Please wait...'):
                X_ligands = filter_pubchem(amw_slider, nar_value, nha_value, fetch=True)
                X_ligands.set_index('CID', inplace=True)
                from_pubchem = True

            st.success(f'Fetched **{count:,}** ligands from PubChem.', icon=check_icon)


# Results
# =======================================
st.header('🔮 Predictions', divider=True)

steps_completed = 0
if X_pocket is not None:
    steps_completed += 1
if X_ligands is not None:
    steps_completed += 1

total_steps = 2
progress = steps_completed / total_steps
st.progress(progress)
st.caption(f"{steps_completed} / {total_steps} steps completed")

if steps_completed < total_steps:
    st.info('Complete all steps to generate predictions.', icon=info_icon)

if X_pocket is not None and X_ligands is not None:
    st.success(
            "Pocket and candidate ligands selected successfully. "
            "Below are the **predictions** of PoLiPaiR in **descending order**.",
            icon=":material/check_circle:"
            )

    # Generate predictions
    with st.spinner('Generating predictions...', show_time=True):
        predictions = predict(X_pocket, X_ligands, from_pubchem)

    predictions.sort_values(by='Score', ascending=False, inplace=True)
    st.dataframe(predictions.iloc[:preds_show_lim], hide_index=True)

    with st.expander('What **Score** represents?'):
        st.info("""
        **Score** represents the predicted probability that the pocket–ligand
        pair is compatible.

        It takes values between 0 and 1, with **higher values indicating better
        fitness**.

        Use it to prioritize ligands for your selected pocket.
        """, icon=info_icon)

# Copyright
# =======================================
st.caption('')
st.caption(':material/copyright: Copyright 2026, Antonios P. Sarikas.')
