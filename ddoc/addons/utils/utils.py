import numpy as np
#import pandas as pd
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
#import matplotlib.patches as patches
import os
import pdb

def study_feature_reconstruct(df, feature_name, label='TARGET', n_features=10):
    df_inliers = df[df[label]==0]
    df_target = df[df[label]==1]
    study_feature(df_target, df_inliers, feature_name, n_features=n_features)

def study_feature(df_target, df_inliers, feature_name, n_features=10):
    '''
    study the behaviour of feature_name depending on target.

    Parameters
    ----------
    n_features: int, default None
        maximal number of values (corresponding to binary features) to plot.
        the most represented values according to both target egal 1 and target egal 0 are
        chosen first. If None, all the values/features are selected/
    '''
    ddf = df_target[feature_name]
    ddi = df_inliers[feature_name]

    # plots for continuous attributes:
    if ddf.dtype == float or ddf.dtype==np.float32:
        # get maximal xrange for target,
        # will be used to have same bin width for the two hist:
        x_range = (min(ddi.min(), ddf.min()), max(ddf.max(), ddi.max()))

        fig, ax = plt.subplots(1, figsize=(20, 20))
        fig.suptitle('%s' %feature_name, fontsize=30)

        # standard plot:
        ax.set_title('standard scale', fontsize=20)
        ddf.plot(kind='hist', normed=True, ax=ax, range=x_range, bins=20,
                 color='#700D35', label='TARGET')
        ddi.plot(kind='hist', normed=True, ax=ax, range=x_range, bins=20,
                 color='#78B62A', alpha=0.5, label='inliers')
        ax.legend(fontsize=20)
        ax.set_ylabel('Frequency', fontsize=20)


    # plot for categorial attributes:
    else:

        countf = ddf.value_counts(normalize=True, dropna=False)
        counti = ddi.value_counts(normalize=True, dropna=False)

        # define n_features as maximal if None:
        if n_features is None:
            n_features = max(len(counti), len(countf))

        # define mask defining most represented features:
        n_features_f = min(n_features, len(countf))
        n_features_i = min(n_features, len(counti))
        countf_argsort = np.argsort(countf)[::-1][:n_features_f]
        counti_argsort = np.argsort(counti)[::-1][:n_features_i]
        mask = list(set(list(countf.index[countf_argsort]) + list(
            counti.index[counti_argsort])))

        # counts data for inliers/outliers:
        # isin do not recognize nan:
        ddi_index_mask = ddi.isin(mask) | ddi.isnull() if np.nan in mask else ddi.isin(mask)
        inliers_counts = ddi[ddi_index_mask].value_counts(normalize=True,
                                                          dropna=False)
        ddf_index_mask = ddf.isin(mask) | ddf.isnull() if np.nan in mask else ddf.isin(mask)
        target1_counts = ddf[ddf_index_mask].value_counts(normalize=True,
                                                        dropna=False)

        # complete unobserved values with 0 counts in such a way
        # inliers_counts and target1_counts share the same axis, which
        # is then the union of their previous supports):
        inliers_counts_, target1_counts_ = inliers_counts.align(target1_counts,
                                                              fill_value=0,
                                                              join='outer')

        # order features in a decreasing order according to inliers:
        order = inliers_counts_.argsort()[::-1]
        inliers_counts_ = inliers_counts_.iloc[order]
        target1_counts_ = target1_counts_.iloc[order]

        # define parameters of the figure:
        reversed_axis = False
        if np.mean([len(str(inliers_counts_.index[i])) for i in range(
                len(inliers_counts_))]) > 10:
            reversed_axis = True
        kind = 'barh' if reversed_axis else 'bar'

        if kind == 'bar':
            fig, ax = plt.subplots(1, figsize=(20, 20),
                                               sharex=True)

            fig.autofmt_xdate(rotation=0)  # do not rotate x axis text
            fig.suptitle(feature_name, size=30)

            ax.set_title('standard scale', fontsize=20)
            inliers_counts_.plot(kind=kind, fontsize=20, label='inliers',
                                 color='#78B62A', ax=ax)
            target1_counts_.plot(kind=kind, fontsize=20, label='TARGET',
                               color='#700D35', width=0.1, ax=ax)
            ax.legend(fontsize=20)

        else:
            fig, ax = plt.subplots(1, figsize=(20, 20), sharey=True)
            fig.suptitle(feature_name, size=35)

            ax.set_title('standard scale', fontsize=20)
            inliers_counts_.plot(kind=kind, fontsize=20, label='inliers',
                                 color='#78B62A', ax=ax)
            target1_counts_.plot(kind=kind, fontsize=20, label='TARGET',
                               color='#700D35', width=0.1, ax=ax)
            ax.legend(fontsize=20)

        return fig
