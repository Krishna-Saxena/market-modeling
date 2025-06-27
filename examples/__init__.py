import itertools
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

from assets.Assets import HeirarchicalAsset
from markets.Markets import Market
from metrics.Metrics import VarType, TimeseriesMetric, StaticMetric

def flip(items, ncol):
    return list(itertools.chain(*[items[i::ncol] for i in range(ncol)]))

def create_market(data_dir: str, market_name: str, timeseries_dir: str, debug: bool=False) -> Market:
    """
    Creates a markets.Markets.Market object from a data directory.

    Args:
        data_dir: The root data directory.
        market_name: The name of the subdirectory of `data_dir/` where the market's data is stored.
        timeseries_dir: The subdirectory of `data_dir/market_name/` where individual assets' timeseries data is stored.
        debug: When true, only the first five assets' timeseries data are added to the returned Market.

    Returns: a Market.
    """
    asset_list = []
    asset_metadata = pd.read_csv(os.path.join(data_dir, market_name, 'asset_metadata.tsv'), sep='\t')

    for i, row in asset_metadata.iterrows():
        if debug:
            if i == 5:
                break

        asset_ts = pd.read_csv(
            os.path.join(data_dir, market_name, timeseries_dir, f"{row['id']}.csv"),
            parse_dates=['date'],
            dtype={'price': np.float32}
        )

        asset = HeirarchicalAsset(
            row['name'],
            row['id'],
            {
                'signal': TimeseriesMetric(asset_ts['date'], asset_ts['price'], VarType.QUANTITATIVE),
                'sector_id': StaticMetric(row['sector'], VarType.CATEGORICAL),
                'industry_id': StaticMetric(row['industry'], VarType.CATEGORICAL)
            }
        )
        asset_list.append(asset)

    return Market(asset_list, market_name)

def plot_train_hist(train_hist: dict):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].plot(train_hist['train_loss'], '--', label='Train')
    ax[0].plot(train_hist['val_loss'], '--', label='Validation')
    ax[0].set(xlabel='Epoch', ylabel='Negative Log Likelihood', title='Loss vs Training Epochs', yscale='log')
    ax[0].legend()

    ax[1].plot([e for e in train_hist['epochs'] if not np.isnan(e)],
             [train_hist['train_MAE'][i] for i, e in enumerate(train_hist['epochs']) if not np.isnan(e)], '--',
             label='Train')
    ax[1].plot([e for e in train_hist['epochs'] if not np.isnan(e)],
             [train_hist['val_MAE'][i] for i, e in enumerate(train_hist['epochs']) if not np.isnan(e)], '--',
             label='Validation')
    ax[1].set(xlabel='Epoch', ylabel='Mean Absolute Error', title='MAE vs Training Epochs')
    ax[1].legend()

    fig.tight_layout()

def plot_covariance_matrix(Sigma: np.array, summ_df: pd.DataFrame):
    # define a linear color map
    disp_Sigma = np.triu(Sigma, k=1)
    max_magn_val = max(np.abs(disp_Sigma.min()), disp_Sigma.max())
    plt.set_cmap('bwr')
    cust_cmap = Normalize(-max_magn_val, max_magn_val)

    fig, ax = plt.subplots(figsize=(10, 10))
    colors = ax.matshow(disp_Sigma, norm=cust_cmap)
    ax.set_xticks(np.arange(disp_Sigma.shape[0]), labels=summ_df['ticker'].values, rotation=90)
    ax.set_yticks(np.arange(disp_Sigma.shape[1]), labels=summ_df['ticker'].values)
    cbar = fig.colorbar(colors, ax=ax, shrink=0.5, aspect=10, pad=0.1, label='', alpha=0.5)
    cbar.set_ticks(np.linspace(cust_cmap.vmin, cust_cmap.vmax, 9))

def plot_predictions(market: Market, sim_dates, sim_results, sim_counterfact):
    """
    Plot predictions on simulation dates.

    Args:
        market: The Market that has been simulated.
        sim_dates: The dates of the simulation.
        sim_results: An array of simulation results, shaped (NUM_SIMS, NUM_ASSETS, len(sim_dates)).
        sim_counterfact: An array of simulation results where no stochasticity has been added, shaped (1, NUM_ASSETS, len(sim_dates)).

    Returns: a plot.
    """
    colorscheme = mpl.colormaps['tab20b']
    fig, ax = plt.subplots(figsize=(10, 8))

    xarr = market.get_dataarray()

    for i in range(colorscheme.N):
        asset_i_id = xarr.isel(ID=i)['ID'].values
        # plot true observations
        ax.plot(
            xarr.time, xarr.sel(ID=asset_i_id, variable='signal'),
            '-o', color=colorscheme.colors[i % colorscheme.N], ms=2,
            label=f'{asset_i_id} signal'
        )
        # plot simulated predictions w/o GBM
        ax.plot(
            sim_dates, sim_counterfact[0, i, :],
            color=colorscheme.colors[i % colorscheme.N], marker='.', linestyle='dashed', ms=5,
            label=f'{asset_i_id} avg pred.'
        )
        # plot simulated predictions w/ GBM
        for j_sim in range(sim_results.shape[0]):
            ax.plot(
                sim_dates, sim_results[j_sim, i, :],
                alpha=0.5, color=colorscheme.colors[i % colorscheme.N], linestyle='dotted'
            )

    ax.set_title('Percent of Total Math SBA Test Score Variance\nExplained by Within-Classroom Variance')
    ax.set(xlabel='Date', ylabel='close price', title='Close Prices vs Time')
    ax.set_xlim([datetime(2025, 1, 1), sim_dates.max()])
    ax.grid(False, 'major', 'x')
    ax.set_yscale('log')
    fig.tight_layout()

    handles, labels = ax.get_legend_handles_labels()
    n_cols = 6
    ax.legend(flip(handles, n_cols), flip(labels, n_cols), loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=n_cols)