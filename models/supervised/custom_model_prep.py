"""
Module: custom_model_prep

This module contains custom modeling preparation and helper classes.

Classes:
- MySplitter: for train/test splitting and separating features and target.
- MyScaler: for log transformation, standard scaling, and min-max scaling.
- MyResults: for evaluating predictions and visualizing performance.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score
)

# Constants used across models (TEST_SIZE can be overriden in MySplitter.train_test)
RSEED = 22
SKEW_THRESHOLD = 0.5
TEST_SIZE = 0.2


class MySplitter:
    """
    Handles train-test splitting and extraction of features (X) and target (y).
    """
    def __init__(self, df):
        self.df = df

    def train_test(self, test_size=TEST_SIZE):
        """
        Split the DataFrame into training and testing sets.
        Displays summary and previews.
        """
        train_df, test_df = train_test_split(self.df, test_size=test_size, random_state=RSEED)

        df_len = len(self.df)
        train_len = len(train_df)
        test_len = len(test_df)

        print(f'Training Data: ({train_len} rows ~{round(train_len / df_len, 2) * 100}%)\n')
        display(train_df.head())
        print(f'Testing Data: ({test_len} rows ~{round(test_len / df_len, 2) * 100}%)\n')
        display(test_df.head())

        return train_df, test_df

    def x_y(self, train_df, test_df):
        """
        Splits train and test DataFrames into X (features) and y (target).
        Prints feature names and class labels.
        """
        train_y = train_df['Label']
        train_X = train_df.drop(['Label'], axis=1)

        test_y = test_df['Label']
        test_X = test_df.drop(['Label'], axis=1)

        features = train_X.columns.values
        classes = ['Low', 'High']

        print(f'Feature Names:\n{features}\n')
        print(f'Class Names:\n{classes}\n')

        return train_X, train_y, test_X, test_y, features, classes


class MyScaler:
    """
    Handles log transformation and scaling methods for numeric features.
    """
    def __init__(self):
        self.scaler_std = StandardScaler()
        self.scaler_minmax = MinMaxScaler()
        
    def log(self, train_X, test_X):
        """
        Log-transforms skewed numeric columns based on a skew threshold.
        """
        transformed_cols = []
        for col in train_X.columns:
            if pd.api.types.is_numeric_dtype(train_X[col]):
                skewness = train_X[col].skew()
                if skewness > SKEW_THRESHOLD:
                    transformed_cols.append(col)
        
        for col in transformed_cols:
            train_X[col] = np.log1p(train_X[col])
            test_X[col] = np.log1p(test_X[col])
            
        print(f'Log transformed (skewness > {SKEW_THRESHOLD}):\n{transformed_cols}\n')
            
        return train_X, test_X
        
    def standard(self, train_X, test_X):
        """
        Applies standard scaling (mean -> 0, variance -> 1).
        """
        train_X = pd.DataFrame(self.scaler_std.fit_transform(train_X), columns=train_X.columns)
        test_X = pd.DataFrame(self.scaler_std.transform(test_X), columns=test_X.columns)
        return train_X, test_X
    
    def minmax(self, train_X, test_X):
        """
        Applies min-max scaling (range -> 0 to 1).
        """
        train_X = pd.DataFrame(self.scaler_minmax.fit_transform(train_X), columns=train_X.columns)
        test_X = pd.DataFrame(self.scaler_minmax.transform(test_X), columns=test_X.columns)
        return train_X, test_X
    
    
class MyResults:
    """
    Stores prediction results and provides methods to print metrics and visualizations.
    """
    def __init__(self, y_true, y_pred, classes):
        self.y_true = y_true
        self.y_pred = y_pred
        self.classes = classes

        self.results_df = pd.DataFrame({
            'Actual': y_true.values,
            'Predicted': y_pred
        }, index=y_true.index)
        self.results_df['Correct'] = self.results_df['Actual'] == self.results_df['Predicted']
        
    def print_metrics(self):
        """
        Prints accuracy, precision, recall, and F1 scores.
        """
        accuracy = accuracy_score(self.y_true, self.y_pred)
        precision = precision_score(self.y_true, self.y_pred)
        recall = recall_score(self.y_true, self.y_pred)
        f1 = f1_score(self.y_true, self.y_pred)

        print(f'\nAccuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall:    {recall:.4f}')
        print(f'F1 Score:  {f1:.4f}')

    def plot_confusion_matrix(self):
        """
        Displays a styled confusion matrix with a custom colormap.
        """
        cm = confusion_matrix(self.y_true, self.y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=self.classes)
        custom_cmap = LinearSegmentedColormap.from_list(
            'black_to_seagreen', ['black', 'seagreen', 'mediumseagreen', 'mediumspringgreen']
        )
        disp.plot(cmap=custom_cmap)
        plt.title('Confusion Matrix')
        plt.show()

    def plot_barcode(self, color_correct='mediumseagreen', color_incorrect='black'):
        """
        Displays a barcode plot representing prediction correctness (green = correct, black = incorrect).
        """
        correctness_map = self.results_df['Correct'].astype(int).values.reshape(1, -1)
        cmap = ListedColormap([color_incorrect, color_correct])
        num_samples = len(self.results_df)
        fig_width = num_samples * 0.01
        fig_height = 0.5

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.imshow(correctness_map, cmap=cmap, interpolation='nearest', aspect='auto')
        ax.axis('off')
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.show()
        