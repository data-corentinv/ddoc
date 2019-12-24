

import argparse
import os
import glob
import subprocess
import signal
import sys
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
sys.path.append('../packages')
sys.path.append('../')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.patches as patches
import pandas.core.algorithms as algos


def compute_results(df_train, thresholds, target, variable):
    """ TO DO COMMENTER
    """
    df_train['constant'] = '0'
    df_train[variable+"_discrete"] = pd.cut(df_train[variable], thresholds)
    result_train = df_train.groupby(variable+"_discrete").agg({"constant" : [len], target : [np.mean]})
    result_train.columns = ["train_size", "train_proportion"]
    result_train = result_train.reset_index()
    return result_train, df_train


def cut_by_quantiles(df_train, target, variable, starting_proportion, infinity, epsilon):
    """
    TO DO COMMENTER
    """
    #df_train = df_train[df_train[variable].notnull()] # Make sure is is na's safe ?
    quantiles = np.linspace(0, 1, int(1/starting_proportion) + 1)
    bins = algos.quantile(df_train[variable], quantiles)
    thresholds, bins_counts = np.unique(bins, return_counts = True)
    large_thresholds = thresholds[bins_counts>1]
    thresholds = np.concatenate([thresholds - epsilon, large_thresholds + epsilon])
    thresholds.sort()
    thresholds[0] = -infinity
    thresholds[-1] = infinity-epsilon/2.0 # Why?
    thresholds = np.append(thresholds, np.array([infinity + epsilon]))

    df = df_train[[variable, target]]

    try:
        df[variable+"_discrete"] = pd.cut(df[variable], thresholds)
    except:

        df[variable+"_discrete"] = 'Warning_Error'
        print('thresholds are not properly handled : '+thresholds)
        # Should probably not happen...

    # This line is useful to clear empty bins created by our epsilon strategy
    thresholds = thresholds[np.concatenate([np.array(df[variable+"_discrete"].value_counts(sort = False)>=1), np.array([True])])]

    df[variable+"_discrete"] = pd.cut(df[variable], thresholds)

    return([df, thresholds])



def plot_numerical_variable(df, target, field, number_of_quantiles, color_rectangle, color_rectangle_nan, axe_fontsize, proportion_minimum_to_display_unique):

    diff_array = np.array(df[field].sort_values())
    try:
        epsilon = np.sort(np.unique(diff_array[1:]-diff_array[:-1]))[1]/4.0
    except: # Only one constant value for the variable
        epsilon = 0.1
        print('There is only one numerical value for variable '+field)


    infinity = max(np.unique(np.abs(df[field])))*1.1+1.0


    if epsilon != epsilon:
        print('There is only one numerical value for variable '+field)
        epsilon = 0.1


    if np.log10(infinity/epsilon)>14:
        # increase epsilon a bit
        print('---------------------------')
        print('Warning : Epsilon and infinity are extremely different : ', epsilon, infinity)

        epsilon = infinity * pow(10, -14)
        print('From now on, epsilon set to : ', epsilon)
        print('---------------------------')
        print(' ')
    # set NaN to +infinity

    if 'montant_trans_' in field or 'nb_trans_' in field:
        df[field] = df[field].fillna(0.0)
    else:
        df[field] = df[field].fillna(infinity)



    df_quant, thresholds = cut_by_quantiles(df,
                                            target,
                                            field,
                                            1.0/number_of_quantiles,
                                            infinity, # infinity
                                            epsilon) # epsilon

    result = compute_results(df, thresholds, target, field)

    N = len(df_quant)
    x_cumsum = np.array(pd.concat([pd.DataFrame([[0]]),result[0]['train_size'].cumsum()]))[:,0]
    y_proportion = result[0]['train_proportion']
    categories = np.array(df_quant[field+'_discrete'].cat.categories)

    values = df_quant.groupby(field+"_discrete").agg({field : [lambda x: x.iloc[0]]})
    values.columns = ["median_value"]
    values = values.reset_index()
    values = np.array(values['median_value'])





    fig, ax = plt.subplots(figsize=(8,4))

    value_to_print_list = []
    value_nan = values == infinity
    for i, value in enumerate(values):
        if value_nan[i]:
            value_to_print_list.append("NaN")
        else:
            value_to_print_list.append(str(value))






    for i in range(len(x_cumsum)-1):
        proportion = y_proportion[i]
        delta_proportion = 3 * np.sqrt(proportion * (1-proportion))/np.sqrt((x_cumsum[i+1] - x_cumsum[i])) # Niveau de risque alpha = 0.002
        # print(proportion)
        # print(delta_proportion)
        if value_to_print_list[i] == 'NaN':
            color_i = color_rectangle_nan
        else:
            color_i = color_rectangle


        ax.add_patch(
            patches.Rectangle(
                (x_cumsum[i], 0),   # (x,y)
                x_cumsum[i+1] - x_cumsum[i],           # width
                100 * proportion,# height
                facecolor=color_i,
                edgecolor  = 'black'
            )
        )
        #plt.annotate(
        #    s='', xy=(0.5 * (x_cumsum[i] + x_cumsum[i+1]),100 * (proportion - delta_proportion)),
        #    xytext=(0.5 * (x_cumsum[i] + x_cumsum[i+1]),100 * (proportion + delta_proportion)),
        #    arrowprops=dict(arrowstyle='-')
        #    )



        if (x_cumsum[i+1] - x_cumsum[i]) > proportion_minimum_to_display_unique * N or value_to_print_list[i] == 'Nan':
            ax.annotate(
                value_to_print_list[i],
                (0.5 * (x_cumsum[i] + x_cumsum[i+1]), 100 * proportion + 2*(max(y_proportion))),
                color='black', fontsize=6, ha='center', va='center')

    ax.set_xlim([0, max(x_cumsum)])
    ax.set_ylim([-5.0 * max(y_proportion)+ 105.0 * 0, 105.0 * max(y_proportion) - 5.0 * 0])
    ax.set_xlabel(field, fontsize=axe_fontsize)
    ax.set_ylabel('Taux de TARGET (%)', fontsize=axe_fontsize)
    major_ticks = np.linspace(0, N, 11)
    ax.set_xticks(major_ticks)
    ax.tick_params(which = 'both', direction = 'out')

    labels = ax.get_xticks().tolist()
    labels = np.percentile(df_quant[field], np.linspace(0,100,11), interpolation = 'nearest')

    labels_with_nan_ok = []
    condition_nan = labels == infinity
    for i, lab in enumerate(labels):
        if condition_nan[i]:
            labels_with_nan_ok.append("NaN")
        else:
            labels_with_nan_ok.append(str(lab))
    ax.set_xticklabels(labels_with_nan_ok)
    #
    plt.tight_layout()
    plt.savefig('temp3.png')
    return(None)


def plot_categorical_variable(df, target, field, color_rectangle, color_rectangle_nan, axe_fontsize, proportion_minimum_to_display_unique):

    # Assume that NaN values have already been replaced by "NaN"
    df['constant'] = '0'
    result_train = df.groupby(field).agg({"constant" : [len], target : [np.mean]})
    result_train.columns = ["train_size", "train_proportion"]
    result_train = result_train.reset_index()

    N = len(df)
    x_cumsum = np.array(pd.concat([pd.DataFrame([[0]]),result_train['train_size'].cumsum()]))[:,0]
    y_proportion = result_train['train_proportion']
    df[field] = df[field].astype('category')
    categories = np.array(df[field].cat.categories)

    values = categories

    fig, ax = plt.subplots(figsize=(8,4))

    for i in range(len(x_cumsum)-1):
        proportion = y_proportion[i]
        delta_proportion = 3 * np.sqrt(proportion * (1-proportion))/np.sqrt((x_cumsum[i+1] - x_cumsum[i])) # Niveau de risque alpha = 0.002
        # print(proportion)
        # print(delta_proportion)
        if values[i] == 'NaN':
            color_i = color_rectangle_nan
        else:
            color_i = color_rectangle


        ax.add_patch(
            patches.Rectangle(
                (x_cumsum[i], 0),   # (x,y)
                x_cumsum[i+1] - x_cumsum[i],           # width
                100 * proportion,# height
                facecolor=color_i,
                edgecolor  = 'black'
            )
        )
        #plt.annotate(
        #    s='', xy=(0.5 * (x_cumsum[i] + x_cumsum[i+1]),100 * (proportion - delta_proportion)),
        #    xytext=(0.5 * (x_cumsum[i] + x_cumsum[i+1]),100 * (proportion + delta_proportion)),
        #    arrowprops=dict(arrowstyle='-')
        #    )



        if (x_cumsum[i+1] - x_cumsum[i]) > proportion_minimum_to_display_unique * N or values[i] == 'Nan':
            ax.annotate(
                values[i],
                (0.5 * (x_cumsum[i] + x_cumsum[i+1]), 100 * proportion + 2.0*(max(y_proportion))),
                color='black', fontsize=6, ha='center', va='center')



    ax.set_xlim([0, max(x_cumsum)])
    ax.set_ylim([-5.0 * max(y_proportion)+ 105.0 * 0, 105.0 * max(y_proportion) - 5.0 * 0])
    ax.set_xlabel(field, fontsize=axe_fontsize)
    ax.set_ylabel('Taux de TARGET (%)', fontsize=axe_fontsize)
    major_ticks = np.linspace(0, N, 11)
    ax.set_xticks(major_ticks)
    ax.tick_params(which = 'both', direction = 'out')

    labels = ax.get_xticks().tolist()

    labels_with_nan_ok = []
    for i, lab in enumerate(labels):
        try:
            labels_with_nan_ok.append(
            categories[(x_cumsum[:-1] <= labels[i]) *  (x_cumsum[1:] > labels[i])][0]
            )
        except:
            labels_with_nan_ok.append(categories[-1]) # Last category

    ax.set_xticklabels(labels_with_nan_ok)
    plt.xticks(rotation=90)
    #
    plt.savefig('temp3.png')

    return(None)
