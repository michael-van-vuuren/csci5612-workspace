"""
Module: custom_data_prep

This module contains custom data preparation functions.

Functions:
- my_corr_plotter: Plots a correlation heatmap with customized styling.
- my_binner: Bins numerical columns into quantile-based integer categories.
- my_count_binner: Bins numerical columns into 5 labeled categories for later vectorization.
- my_count_vectorizer: Converts binned labels into feature counts per row for count-based models.
- my_bin_plotter: Visualizes the distribution of binned values across columns.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns

# Set default plotting resolution and figure size
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [4, 3]


def my_corr_plotter(df, inner_fontsize=6, outer_fontsize=6):
    """
    Plot a correlation heatmap with customized styling.
    """
    custom_cmap = LinearSegmentedColormap.from_list(
        'black_seagreen_diverge', ['black', 'white', 'mediumseagreen']
    )
    ax = sns.heatmap(df.corr(numeric_only=True), annot=True, cmap=custom_cmap, fmt='.2f', vmin=-1, vmax=1, center=0, 
                annot_kws={'size': inner_fontsize, 'weight': 'semibold'})
    plt.xticks(rotation=90)
    plt.xticks(fontsize=outer_fontsize)
    plt.yticks(fontsize=outer_fontsize)
    
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=outer_fontsize)
    
    plt.show()

    
def my_binner(df):
    """
    Bin numerical columns using quantiles (qcut), returning integer labels.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    binned_df = pd.DataFrame(index=df.index)

    for col in numeric_df.columns:
        col_data = numeric_df[col].dropna()
        n = len(col_data)

        if n < 2:
            num_bins = 1
        else:
            # Use a modified Sturges' Rule to determine number of bins
            factor = 0.8
            num_bins = int(np.ceil(factor * (1 + np.log2(n))))
            num_bins = max(2, num_bins)

        unique_vals = col_data.nunique()
        actual_bins = min(num_bins, unique_vals)

        if actual_bins < 2:
            binned_series = pd.Series(0, index=col_data.index, dtype='Int64')
            binned_df[col] = binned_series.reindex(df.index)
        else:
            # Apply quantile binning with fallback for dropped duplicates
            binned_series = pd.qcut(numeric_df[col], q=actual_bins, labels=False, duplicates='drop')
            binned_df[col] = binned_series.astype('Int64')

    return binned_df


def my_count_binner(df):
    """
    Bin numerical columns into 5 bins using labels 
    'low', 'low-medium', 'medium', 'medium-high', 'high'
    to prepare for my_count_vectorizer.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    binned_df = pd.DataFrame(index=df.index)

    for col in numeric_df.columns:
        col_data = numeric_df[col].dropna()
        n = len(col_data)

        if col_data.nunique() < 1:
            binned_series = pd.Series('medium', index=col_data.index, dtype='string')
            binned_df[col] = binned_series.reindex(df.index)
        else:
            try:
                # Try quantile binning with categorical labels
                binned_series = pd.qcut(
                    col_data, 
                    q=5, 
                    labels=['low', 'low-medium', 'medium', 'medium-high', 'high'], 
                    duplicates='drop'
                )
            except ValueError:
                try:
                    # Fallback to equal-width binning if qcut fails
                    binned_series = pd.cut(
                        col_data, 
                        bins=5, 
                        labels=['low', 'low-medium', 'medium', 'medium-high', 'high']
                    )
                except ValueError:
                    # If binning still fails, assign default label of 'low'
                    binned_series = pd.Series('low', index=col_data.index, dtype='string')

            binned_df[col] = binned_series.astype('string').reindex(df.index)

    return binned_df


def my_count_vectorizer(df_binned, label_col='Label'):
    """
    Convert binned categorical values into feature counts per label.
    """
    binned_only = df_binned.drop(columns=[label_col])

    # Count how many times each bin label appears in each row
    collapsed_df = pd.DataFrame({
        'low': (binned_only == 'low').sum(axis=1).astype('Int64'),
        'low-medium': (binned_only == 'low-medium').sum(axis=1).astype('Int64'),
        'medium': (binned_only == 'medium').sum(axis=1).astype('Int64'),
        'medium-high': (binned_only == 'medium-high').sum(axis=1).astype('Int64'),
        'high': (binned_only == 'high').sum(axis=1).astype('Int64'),
        label_col: df_binned[label_col]
    })

    return collapsed_df


def my_bin_plotter(df):
    """
    Plot bar charts of binned value counts for each column
    to ensure roughly uniform distribution of bins.
    
    Compatible with output from both my_binner and my_count_vectorizer.
    """
    value_counts_dict = {}

    ordered_bins = ['low', 'low-medium', 'medium', 'medium-high', 'high']

    for col in df.columns:
        counts = df[col].value_counts()

        if df[col].dtype == 'string':
            counts = counts.reindex(ordered_bins, fill_value=0)

        elif pd.api.types.is_integer_dtype(df[col]) or pd.api.types.is_float_dtype(df[col]):
            counts = counts.sort_index()

        value_counts_dict[col] = counts

    num_cols_to_plot = len(df.columns)
    ncols_grid = 3
    nrows_grid = int(np.ceil(num_cols_to_plot / ncols_grid))
    fig, axes = plt.subplots(nrows=nrows_grid, ncols=ncols_grid, figsize=(14, 4 * nrows_grid))
    axes = axes.flatten()

    for i, (col_name, counts) in enumerate(value_counts_dict.items()):
        ax = axes[i]
        x = counts.index.tolist()
        y = counts.values.tolist()
        ax.bar(x, y, color='mediumseagreen')
        ax.set_title(col_name, fontsize=14, fontweight='semibold')
        ax.set_xlabel('Binned Value', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_xticks(range(len(x)))
        ax.set_xticklabels(x, ha='center', fontsize=11)
        ax.tick_params(axis='x', labelsize=9)
        ax.tick_params(axis='y', labelsize=11)

    if len(value_counts_dict) < len(axes):
        for j in range(len(value_counts_dict), len(axes)):
            fig.delaxes(axes[j])

    fig.tight_layout()
    plt.show()
    