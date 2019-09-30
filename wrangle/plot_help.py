import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.gridspec import GridSpec
import seaborn as sns
import numpy as np



def plot_hist(series, start=0, end=6, step=0.5, xlabel='xlabel', ylabel='count', feature='title', saveloc=None, density=False, fit=False, orientation='vertical'):
    """
    helper function for plotting histogram
    series = pandas serries
    start = bin start position
    end = bin end
    step = step size between bins
    xlabel = x label title
    ylabel = y label title
    feature = name of feature 
    saveloc = location to save to
    
    """
    plt.hist(series, bins=np.arange(start, end, step), density=density)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    #stats
    plt.axvline(series.median(), color='red',linestyle='solid', linewidth=1, label='median')
    plt.axvline(series.mean(), color='green',linestyle='solid', linewidth=1, label='mean')
    plt.axvline(series.describe()['25%'], color='orange',linestyle='dashed', linewidth=1, label='Q1')
    plt.axvline(series.describe()['75%'], color='orange',linestyle='dashed', linewidth=1, label='Q3')
    
    #Add a fit
    if fit:
        y = mlab.normpdf(np.arange(start, end, step), series.median(), series.std())
        l = plt.plot(np.arange(start, end, step), y, 'r--', linewidth=1)
    
    
    plt.legend()
    plt.title("Yelp {} Distribution".format(feature))
    plt.tight_layout()
    
    if saveloc:
        plt.savefig(saveloc)
    plt.show()
    
    
    
def df_plot_count(df, col_drop, df_name,k=20,savepath=None):
    """
    plots the count of top values in a 1/0 time dataframe
    df = pandas dataframe
    col_drop = list of string name of columns to drop
    savepath = path to save to
    """
    
    #sum up by column
    val_count = df.drop(columns=col_drop).sum(axis=0).sort_values(ascending=False)
    
    #avoid error for slicing beyond index
    if len(val_count)<k:
        k = len(val_count)
    
    top_k_valcount = val_count[0:k]
    
    
    plt.figure(figsize=(15,5))
    sns.barplot(top_k_valcount.index, top_k_valcount.values, alpha=0.8, order=top_k_valcount.index)
    plt.title('Yelp Top {} Features for {}'.format(k, df_name))
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel(df_name, fontsize=12)
    plt.xticks(rotation=90)
    
    if savepath:
        plt.savefig(savepath)
    
    plt.show()
    
    
def plot_count_top_k(df, col, k=20,savepath=None):
    """
    plots the top k value counts in a dataframe
    df = pandas dataframe
    col = string name of columns   
    k = top k integer values
    savepath = path to save to
    """
    
    val_count = df[col].value_counts()
    
    #avoid error for slicing beyond index
    if len(val_count)<k:
        k = len(val_count)
    
    top_k_valcount = val_count[0:k]
    
    
    plt.figure(figsize=(15,5))
    sns.barplot(top_k_valcount.index, top_k_valcount.values, alpha=0.8, order=top_k_valcount.index)
    plt.title('Yelp top {} {}'.format(k, col))
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel(col, fontsize=12)
    plt.xticks(rotation=90)
    
    if savepath:
        plt.savefig(savepath)
    
    plt.show()
    
    
def df_binary_corr_plot(df, col_drop, k=20, title=None,savepath=None, fig_size = (15,15)):
    """
    Takes in a dataframe and plots the correlation matrix for top k columns based on feature presence
    df = pandas dataframe
    col_drop = columns to drop
    k = top k
    savepath = save directory
    fig_size = tuple for figure size
    """
    
    #sum  over rows minus dropped cols
    cats_sum = df.drop(columns=col_drop).sum(axis=0)
    
    #sort descending
    cats_sum_sorted = cats_sum.sort_values(ascending=False)
    
    #get top k column names
    cats_sum_topk = cats_sum_sorted.index[0:k]
    
    df_corr = df[cats_sum_topk].corr()
    
    # Generate a mask for the upper triangle
    mask = np.zeros_like(df_corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=fig_size)

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(df_corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

    if title:
        plt.title("{} Pearson Correlation Heatmap".format(title))

    if savepath:
        plt.savefig("plots/portfolio_corr.png")
    
    plt.show()
    
    
    