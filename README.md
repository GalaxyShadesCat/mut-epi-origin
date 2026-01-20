## Conda environment

Activate (first time):

```bash
source /home/lem/miniconda3/etc/profile.d/conda.sh
conda activate mut-epi-origin
```

Export the environment (for another machine):

```bash
conda env export -n mut-epi-origin --no-builds > mut-epi-origin.yml
```

Recreate elsewhere:

```bash
conda env create -f mut-epi-origin.yml
conda activate mut-epi-origin
```

If `conda activate` fails, run `conda init` once and restart your shell.

## Streamlit apps

Results explorer:

```bash
streamlit run tools/results_dashboard.py
```

Track visualisation:

```bash
streamlit run tools/track_visualisation.py
```

Track visualisation assets live in `tools/track_visualisation_assets/`.
