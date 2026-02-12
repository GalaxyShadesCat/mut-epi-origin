## Conda environment

Create the environment:

```bash
conda env create -f environment.yml
conda activate mut-epi-origin
pip install -r requirements.txt
```

If `conda activate` fails, run `conda init` once and restart your shell.

## Streamlit apps

Track Visualisation:

```bash
streamlit run tools/track_visualisation.py
```

Results Dashboard:

```bash
streamlit run tools/results_dashboard/run.py
```
