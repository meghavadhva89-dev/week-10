# Week 10

This week's lab is meant to introduce you to the idea of Supervised Learning. In particular, we'll cover the following topics:

- Regression vs. Classification
- Model Evaluation
- Visualizing Models

## Setup

1. [Fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo#forking-a-repository) this repository.
2. [Create a Codespace](https://docs.github.com/en/codespaces/developing-in-a-codespace/creating-a-codespace-for-a-repository#creating-a-codespace-for-a-repository) for your repository. Use this to view the lab notebook and work on your weekly coding exercise.
3. Once you're ready, [commit and push](https://docs.github.com/en/codespaces/developing-in-a-codespace/using-source-control-in-your-codespace#committing-your-changes) your final changes to your repository. *Note: You can also make quick edits using the [GitHub Dev Editor](https://docs.github.com/en/codespaces/the-githubdev-web-based-editor#opening-the-githubdev-editor).*

## Packages Available:

The environment for this week is built with the following environment.yml:

```yml
name: h501-week-10
dependencies:
  - python=3.11
  - pip
  - pip:
    - ipykernel
    - pandas
    - numpy
    - seaborn
    - streamlit
    - scikit-learn
    - plotly
```

*Note: you are welcome to install more packages in your codespace, but they will not be used by the autograder.*

## Files of interest

- `train.py` — trains the models and writes pickles:
  - `model_1.pickle` — LinearRegression trained on `100g_USD` → `rating`
  - `model_2.pickle` — DecisionTreeRegressor (estimator). A `roast_map.pickle` file is also written for the categorical mapping.
  - `model_3.pickle` — a dict containing `{"model", "vectorizer"}` for the TF‑IDF text model
- `apputil.py` — provides `predict_rating(df_X, text=False)` used by the notebook/app. Supports both numeric and text prediction modes.
- `test_load_model.py` — a small smoke test that demonstrates predictions (optional; can be removed for source-only submission).
- `exercises.ipynb` — the lab notebook; includes Plotly visualizations appended for model exploration.

## Repro / Common commands (PowerShell)

Train models (creates/overwrites `model_*.pickle` files):
```powershell
py -3 train.py
```

Run the smoke test (uses existing pickles):
```powershell
py -3 test_load_model.py
```

Lint / style check:
```powershell
py -3 -m flake8 --max-line-length=88
```

Open and run the notebook in VS Code or Jupyter. The notebook's final cells contain interactive Plotly visualizations (price vs rating, rating by roast, residuals, feature importances, TF‑IDF top terms).

## Submission notes

- This repository has been prepared for a source-only submission: generated model
  pickles and HTML visualization artifacts have been removed.

- If the autograder expects pickled models, re-create them by running:

```powershell
py -3 train.py
```

- If you want a minimal submission (no tests), you can remove `test_load_model.py` (if present).

## Next steps / optional

- I can remove the pickles and/or `test_load_model.py` for a source-only submission.
- I can also add a tiny pytest file that checks `predict_rating` on a small synthetic input (happy path + unknown roast). If you'd like either, tell me which and I'll add it.
