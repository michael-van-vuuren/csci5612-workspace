import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [4, 3]


def my_corr_plotter(df, inner_fontsize=6, outer_fontsize=6):
    custom_cmap = LinearSegmentedColormap.from_list(
        'black_seagreen_diverge', ['black', 'white', 'mediumseagreen']
    )
    ax = sns.heatmap(df.corr(numeric_only=True), annot=True, cmap=custom_cmap, fmt='.2f', vmin=-1, vmax=1, center=0, 
                annot_kws={'size': inner_fontsize})
    plt.xticks(rotation=90)
    plt.xticks(fontsize=outer_fontsize)
    plt.yticks(fontsize=outer_fontsize)
    
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=outer_fontsize)
    
    plt.show()

    
def my_binner(df):
    numeric_df = df.select_dtypes(include=[np.number])
    binned_df = pd.DataFrame(index=df.index)

    for col in numeric_df.columns:
        col_data = numeric_df[col].dropna()
        n = len(col_data)

        if n < 2:
            num_bins = 1
        else:
            # Sturge's Rule: k = 1 + log2(N)
            factor = 0.8
            num_bins = int(np.ceil(factor * (1 + np.log2(n))))
            num_bins = max(2, num_bins)

        unique_vals = col_data.nunique()
        actual_bins = min(num_bins, unique_vals)

        if actual_bins < 2:
            binned_series = pd.Series(0, index=col_data.index, dtype='Int64')
            binned_df[col] = binned_series.reindex(df.index)
        else:
            binned_series = pd.qcut(numeric_df[col], q=actual_bins, labels=False, duplicates='drop')
            binned_df[col] = binned_series.astype('Int64')

    return binned_df

def my_count_binner(df):
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
                binned_series = pd.qcut(
                    col_data, 
                    q=5, 
                    labels=['low', 'low-medium', 'medium', 'medium-high', 'high'], 
                    duplicates='drop'
                )
            except ValueError:
                try:
                    binned_series = pd.cut(
                        col_data, 
                        bins=5, 
                        labels=['low', 'low-medium', 'medium', 'medium-high', 'high']
                    )
                except ValueError:
                    binned_series = pd.Series('low', index=col_data.index, dtype='string')

            binned_df[col] = binned_series.astype('string').reindex(df.index)

    return binned_df

def my_count_vectorizer(df_binned, label_col='Label'):
    binned_only = df_binned.drop(columns=[label_col])

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
        ax.set_title(col_name)
        ax.set_xlabel('Binned Value')
        ax.set_ylabel('Count')
        ax.set_xticks(range(len(x)))
        ax.set_xticklabels(x, ha='center')
        ax.tick_params(axis='x', labelsize=9)

    if len(value_counts_dict) < len(axes):
        for j in range(len(value_counts_dict), len(axes)):
            fig.delaxes(axes[j])

    fig.tight_layout()
    plt.show()