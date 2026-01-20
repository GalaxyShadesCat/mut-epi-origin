## Conda environment

Create the environment:

```bash
conda env create -f environment.yml
conda activate mut-epi-origin
```

If `conda activate` fails, run `conda init` once and restart your shell.

## Streamlit apps

Track visualisation:

```bash
streamlit run tools/track_visualisation.py
```

Track visualisation assets live in `tools/track_visualisation_assets/`.

Results explorer:

```bash
streamlit run tools/results_dashboard.py
```
