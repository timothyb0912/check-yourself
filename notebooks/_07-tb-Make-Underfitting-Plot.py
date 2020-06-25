# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import sys
from collections import OrderedDict
import scipy.stats
import numpy as np
import pandas as pd
import pylogit as pl
import matplotlib.pyplot as plt

sys.path.insert(0, '../src')
from visualization import predictive_viz as viz

# %matplotlib inline
# -

# # Generate fake data

# +
np.random.seed(1019)

num_obs = 500
x_1 = scipy.stats.gamma.rvs(a=4, scale=0.5, size=num_obs)

def logistic(x):
    v = -2 + 2 * x - x**2
    neg_v = np.clip(-1 * v, -1e700, 1e300)
    return 1 / (1 + np.exp(neg_v))

y_probs_1 = logistic(x_1)

x = np.concatenate((x_1[:, None], np.zeros(num_obs)[:, None]),
                   axis=1).ravel()

y_probs = np.concatenate((y_probs_1[:, None], (1 - y_probs_1)[:, None]),
                   axis=1).ravel()

obs_ids = np.repeat(np.arange(num_obs) + 1 , 2)

y = viz.simulate_choice_vector(y_probs, obs_ids).ravel()

df = pd.DataFrame({'obs_id': obs_ids,
                   'alt_id': np.tile(np.array([1, 2]), num_obs),
                   'x': x, 'sin_x':np.sin(x),
                   'x2':x**2, 'x3':x**3,
                   'x4':x**4, 'x5':x**5,
                   'y': y})

# -

# # Generate correct and incorrect specifications

# +
bad_spec = OrderedDict()
bad_names = OrderedDict()

bad_spec['intercept'] = [1]
bad_names['intercept'] = ['intercept']

bad_spec['x'] = 'all_same'
bad_names['x'] = 'x'

good_spec = OrderedDict()
good_names = OrderedDict()

good_spec['intercept'] = [1]
good_names['intercept'] = ['intercept']

good_spec['x'] = 'all_same'
good_names['x'] = 'x'

good_spec['x2'] = 'all_same'
good_names['x2'] = 'x2'

overfit_spec = OrderedDict()
overfit_names = OrderedDict()

overfit_spec['intercept'] = [1]
overfit_names['intercept'] = ['intercept']

overfit_spec['x'] = 'all_same'
overfit_names['x'] = 'x'

overfit_spec['sin_x'] = 'all_same'
overfit_names['sin_x'] = 'sin_x'

overfit_spec['x3'] = 'all_same'
overfit_names['x3'] = 'x3'

overfit_spec['x5'] = 'all_same'
overfit_names['x5'] = 'x5'

# -

# Estimate both models
bad_mnl = pl.create_choice_model(df,
                                 'alt_id',
                                 'obs_id',
                                 'y',
                                 bad_spec,
                                 model_type='MNL',
                                 names=bad_names)
bad_mnl.fit_mle(np.zeros(len(bad_names)), method='bfgs')
bad_mnl.get_statsmodels_summary()

# Estimate both models
good_mnl = pl.create_choice_model(df,
                                 'alt_id',
                                 'obs_id',
                                 'y',
                                 good_spec,
                                 model_type='MNL',
                                 names=good_names)
good_mnl.fit_mle(np.zeros(len(good_names)), method='bfgs')
good_mnl.get_statsmodels_summary()

# Estimate both models
overfit_mnl = pl.create_choice_model(df,
                                 'alt_id',
                                 'obs_id',
                                 'y',
                                 overfit_spec,
                                 model_type='MNL',
                                 names=overfit_names)
overfit_mnl.fit_mle(np.zeros(len(overfit_names)), method='bfgs')
overfit_mnl.get_statsmodels_summary()

# +
# Get the probabilities of y = 1 according to the three models
# and order the probabilities according to increasing x-values
alt_1_rows = np.where((df['alt_id'] == 1).values)[0]
alt_1_order = np.argsort(x_1)

p_underfit = bad_mnl.long_fitted_probs[alt_1_rows][alt_1_order]
p_true = good_mnl.long_fitted_probs[alt_1_rows][alt_1_order]
p_overfit = overfit_mnl.long_fitted_probs[alt_1_rows][alt_1_order]
# -

p_true.mean(), p_underfit.mean(), p_overfit.mean()

# +
x_line = x_1[alt_1_order]

overfit_color =\
    (0.984313725490196, 0.6039215686274509, 0.6)
# underfit_color = '#a6bddb'
underfit_color =\
    (0.792156862745098, 0.6980392156862745, 0.8392156862745098)

fig, ax = plt.subplots(1, figsize=(10, 6))
ax.plot(x_line, p_underfit, linestyle='--',
        c=underfit_color, label='Underfit')
ax.plot(x_line, p_true, c='#045a8d', label='True')
ax.plot(x_line, p_overfit, linestyle='-.',
        c=overfit_color, label='Overfit')

# ax.set_xlabel('Bicycle Travel Distance (miles)', fontsize=12)
# ax.set_ylabel('Probability\nof Bicycling',
#               rotation=0, labelpad=40, fontsize=12)
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('P(Y = 1 | X)',
              rotation=0, labelpad=40, fontsize=12)

ax.legend(loc='best')

fig.tight_layout()
fig.savefig('../reports/figures/underfitting_example.pdf',
            dpi=500, bbox_inches='tight')
