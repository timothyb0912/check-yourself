Check Yourself Before You Wreck Yourself: Assessing Discrete Choice Models Through Predictive Simulations
==============================

A case study in discrete choice model assessment with predictive simulations.

This repository contains the replication data and code for

    Brathwaite, Timothy. "Check yourself before you wreck yourself: Assessing
    discrete choice models through predictive simulations" arXiv preprint
    arXiv:1806.02307 (2018). https://arxiv.org/abs/1806.02307.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or
    │                         `make train`
    │
    ├── README.md          <- The top-level README for developers using this
    │                         project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for
    │                         details
    │
    ├── models             <- Trained and serialized models, model predictions,
    │                         or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number
    │                         (for ordering), the creator's initials, and a
    │                         short `-` delimited description, e.g.
    │                         `_01-jqp-initial-data-exploration`
    │
    ├── references         <- Data dictionaries, manuals, key reference papers,
    │                         and all other explanatory materials
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in
    │                         reporting
    │   └── tables         <- LaTex files for tables to be used in reporting
    │   └── complete       <- LaTex files for the final report and journal
    │                         submission
    │
    ├── requirements.txt   <- The requirements file to reproduce the analysis
    │                         environment, e.g. generated with
    │                         `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .)
    │                         so src can be imported
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── get_car_data.R
    │   │   └── convert_car_data_from_wide_to_long.py
    │   │
    │   ├── features       <- Scripts to turn raw data into modeling features
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained
    │   │   │                 models to make predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results
    │                         oriented visualizations
    │       └── predictive_viz.py
    │
    └── tox.ini            <- tox settings file; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
