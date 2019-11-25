import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.gridspec import GridSpec
import seaborn as sns
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve
from sklearn.utils.multiclass import unique_labels


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
    
    plt.tight_layout()
    
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
    
    plt.tight_layout()
    
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
        

    plt.tight_layout()
    
    if savepath:
        plt.savefig("plots/portfolio_corr.png")
    
    plt.show()
    
    
    
def find_weekend_indices(datetime_array, weekend=5):
    """
    Returns all indeces of Saturdays & Sundays in a datetime array
    datetime_array(pandas) = pandas datetime array
    weekend(int) = assume weekend starts at day=5=Saturday
    """
    #empty list to tore indeces
    indices = []
    
    
    for i in range(len(datetime_array)):
        #  get day of the week with Monday=0, Saturday=5, Sunday=6
        if datetime_array[i].weekday() >= weekend:
            indices.append(i)
            
    return indices


def highlight_datetimes(indices, ax, df, facecolor='green', alpha_span=0.2):
    """
    Highlights all weekends in an axes object
    indices(list) = list of Saturdays and Sundays indeces corresponding to dataframe
    ax(matplot) = pyplot object
    df(pandas) = pandas dataframe
    """
    i = 0
    #iterate over indeces
    while i < len(indices)-1:
        #highlight from i to i+1
        ax.axvspan(df.index[indices[i]], 
                   df.index[indices[i] + 1], 
                   facecolor=facecolor, 
                   edgecolor='none', 
                   alpha=alpha_span)
        i += 1

        
def plot_datetime(df, title="Yelp Businesses Checkins", 
                  highlight=True, 
                  weekend=5, 
                  ylabel="Number of Visits",
                  saveloc=None, 
                  facecolor='green', 
                  alpha_span=0.2):
    """
    Draw a plot of a dataframe that has datetime object as its index
    df(pandas) = pandas dataframe with datetime as indeces
    highlight(bool) = to highlight or not
    title(string) = title of plot
    saveloc(string) = where to save file
    """
    
    #instantiate fig and ax object
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True,figsize=(15,5))

    #draw all columns of dataframe
    for v in df.columns.tolist():
        axes.plot(df[v], label=v, alpha=.8)
        
    if highlight:
        #find weekend indeces
        weekend_indices = find_weekend_indices(df.index, weekend=5)
        #highlight weekends
        highlight_datetimes(weekend_indices, axes, df, facecolor)

    #set title and y label
    axes.set_title(title, fontsize=12)
    axes.set_ylabel(ylabel)
    axes.legend()
    plt.tight_layout()
    
    #add xaxis gridlines
    axes.xaxis.grid(b=True, which='major', color='black', linestyle='--', alpha=1) 

    plt.tight_layout()
    
    #savefig if
    if saveloc:
        fig.savefig(saveloc)
    plt.show()
    
    
def plot_confusion_matrix(y_true,
                          y_pred,
                          classes=['closed', 'open'],
                          normalize=False,
                          title='classifier',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title='CM for {}'.format(title),
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.show()
    plt.show()

def plot_roc_curve(y_test, y_pred, classifier_name='Classifier', saveloc=None):
    """
    Plot the ROC curve for a single classifier on a binary class and print ROC score
    y_test(np.array) = ground truth numpy array
    y_pred(np.array) = prediction
    classifier_name(string) = name of classifier
    saveloc(string) = where to save
    
    """
    
    #compute ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    
    #plot and decorate
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.title("ROC Curve for {}".format(classifier_name))
    plt.xlabel("FPR")
    plt.ylabel("TPR (recall)")
    
    
    #find threshold closes to 0
    close_zero = np.argmin(np.abs(thresholds))
    plt.plot(fpr[close_zero], tpr[close_zero], 'o'
             , markersize=10, label='threshold zero', fillstyle=None, c='k', mew=2)
    
    plt.legend(loc=4)
    plt.tight_layout()
    
    if saveloc:
        plt.savefig(saveloc)
    
    plt.show()
    
    score = roc_auc_score(y_test, y_pred, average='weighted')
    print("Area Under Curve = {:.4f}".format(score))
    
def plot_prc_curve(y_test, probas_pred, classifier_name='Classifier', saveloc=None):
    precision, recall, thresholds = precision_recall_curve(y_test, probas_pred)
    # find threshold closest to zero
    close_zero = np.argmin(np.abs(thresholds)) 
    
    plt.plot(precision[close_zero], recall[close_zero], 'o', markersize=10,
             label="threshold zero", fillstyle="none", c='k', mew=2)

    plt.plot(precision, recall, label="precision recall curve")
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.title("Precision-Recall for {}".format(classifier_name))
    plt.tight_layout()
    
    if saveloc:
        plt.savefig(saveloc)
    
    plt.show()
    
def report_stats(y_test, y_pred, probas_pred, classifier_name='Classifier', saveloc=None):
    """
    Summary of statistics for prediction
    
    """
    
    print("Classification Report")
    print(classification_report(y_test, y_pred))
    
    print(15*'-')
    
    plot_confusion_matrix(y_test,
                          y_pred,
                          classes=['closed', 'open'],
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues)
    
    plot_roc_curve(y_test, y_pred, classifier_name='Classifier', saveloc=None)
    plot_prc_curve(y_test, probas_pred, classifier_name='Classifier', saveloc=None)
    
    
    