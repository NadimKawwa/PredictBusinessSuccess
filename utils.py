import os, glob, pickle
import pandas as pd
import numpy as np
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler



def chunk_loader(directory, c_size=100_000, orient='columns', lines=True, read_limit=0, index_col=0):
    """
    Reads a directory in chunks, infers type if json or csv format
    directory(str) = location of file
    c_size(int) = size of individual chunk
    oritnet(str) = orientation of expected format
    read_limit(int) = limit of file to read
    index_col(int) = Column(s) to use as the row labels
    """
    
    if '.json' in directory:
        #return JsonReader object
        review_df_chunk = pd.read_json(directory, orient=orient,lines=lines, chunksize=c_size)
    elif '.csv' in directory:
        review_df_chunk = pd.read_csv(directory, chunksize=c_size, index_col=index_col)
    else:
        print('unsopported file type, must be json or csv')
        return None
        
    #keep track of file limit
    limit=0
    
    #store chunks in list
    chunk_list = []
    
    #loop over chunks
    for df in review_df_chunk:
        #add to limit
        limit += df.shape[0]
        #append to list
        chunk_list.append(df)
        
        #check if a limit was set
        if read_limit>0:
            if limit >=read_limit:
                df= pd.concat(chunk_list)
                return df.loc[:read_limit-1,:]
        
    #combine results in one dataframe
    df= pd.concat(chunk_list)
    
    
    return df


def reduce_merge(df_list, key):
    """
    merge a list dataframes based on a shared key
    """
    #apply function of two arguments cumulatively to the items of iterable
    df_merge = reduce(lambda left, right: pd.merge(left, right, on=key), df_list)
    
    return df_merge
        
def train_test_scale (df, target, random_state=None):
    """
    preprocess data
    df(pandas) = dataframe
    target(str) = name of target column
    
    """
    
    X = df.drop(columns=[target])
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2, 
                                                        random_state = random_state, 
                                                        stratify=y)
    #make np.array
    #note method to_numpy() might not work depending on version
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    #y_train.reset_index(inplace=True, drop=True)
    #y_test.reset_index(inplace=True, drop=True)
    
    #instantiate scaler
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    
    return scaler.transform(X_train), scaler.transform(X_test), y_train, y_test

def make_num_df (df, drop_cols=['latitude', 'longitude', 'postal_code']):
    """
    Drops columns and returns numerical entries from pandas dataframe
    df(pandas): dataframe
    drop_cols(list): array of columns to drop, deemed noisy and/or of little use
    """
    
    df_drop = df.drop(columns=drop_cols)
    
    #make dataframe of numeric types
    df_num = df_drop.select_dtypes(include=[np.float64, np.int64]).copy()
    
    return df_num




def binary_get_minority(y):
    """
    Identify the minority and majority class in a 1-dimensional array
    y(array): array containing target
    
    """
    #return the sorted unique elements of an array
    unique, counts = np.unique(y, return_counts=True)
    
    #check target is binary and 1-D
    if (len(unique) != 2) or (np.ndim(y)!= 1):
        print("Target must be binary and 1-dimensional... Returning None")
        return None
    
    if counts[0]<counts[1]:
        minority, majority = unique[0], unique[1]
    elif counts[0]==counts[1]:
        print("array is balanced... Returning classes as is")
        return unique[0], unique[1]
    else:
        minority, majority = unique[1], unique[0]
        
    return minority, majority


def split_major_minor(X, y):
    """
    Split feature and target arrays into majority and minority
    X(numpy): feature space
    y(numpy): targets
    """
    
    #identify minorit and majority
    minority, majority = binary_get_minority(y)
    
    #get indeces of minority
    minority_index = np.where(y==minority)
    #indeces of majority
    majority_index = np.where(y==majority)

    #get minority and majority features and targets
    y_minority, y_majority = y[minority_index].copy(), y[majority_index].copy()

    X_minority, X_majority = X[minority_index].copy(), X[majority_index].copy()
    
    return X_minority, X_majority, y_minority, y_majority



def parallel_permute(X, y):
    """
    Permute two arrays X and y in unision along first dimension
    X(array): n-dimensional array
    y(array): n-dimensional array
    """
    #check lengths are same
    assert len(X)== len(y)
    
    #create permuted index
    perm = np.random.permutation(len(y))

    #return permuted array to maintain randomness
    X_perm = X[perm]
    y_perm = y[perm]
    
    return X_perm, y_perm




def undersample(X, y):
    """
    Return balanced features and targets by undersampling
    X(numpy): feature array
    y(numpy): targets array
    """
    X_minority, X_majority, y_minority, y_majority = split_major_minor(X, y)
    
    #draw random indeces from majority and limit by number of entries in minority
    rand_idx = np.random.choice(len(y_majority), 
                                len(y_minority))

    #apply undersampling
    X_majority_under = X_majority[rand_idx]
    y_majority_under = y_majority[rand_idx]

    #check lengths
    assert len(X_majority_under) == len(X_minority)
    assert len(y_majority_under) == len(y_minority)


    #vertically stack
    X_under = np.concatenate((X_majority_under, X_minority), axis=0)
    y_under = np.concatenate((y_majority_under, y_minority), axis=0)

    X_under, y_under = parallel_permute(X_under, y_under)
    
    return X_under, y_under




def oversample(X, y):
    #get major and minor
    X_minority, X_majority, y_minority, y_majority = split_major_minor(X, y)
    
    #get sizing different
    n_copy = len(X_majority) / len(X_minority)
    #get integer portion
    n_copy_int = int(n_copy)
    #get fraction portion
    n_copy_frac = n_copy % 1
    
    #replicate minority by integer portion
    X_minority_over = np.repeat(X_minority, n_copy_int, axis=0)
    y_minority_over = np.repeat(y_minority, n_copy_int, axis=0)

    #replicate minority by fraction
    rand_idx = np.random.choice(len(y_minority), int(n_copy_frac*len(y_majority)))
    X_over_frac = X_minority[rand_idx]
    y_over_frac = y_minority[rand_idx]
    
    #concatenate to create oversampled minority
    X_minority_over = np.concatenate((X_minority_over, X_over_frac), axis=0)
    y_minority_over = np.concatenate((y_minority_over, y_over_frac), axis=0)
    
    #concatenate with majority class
    X_over = np.concatenate((X_minority_over, X_majority), axis=0)
    y_over = np.concatenate((y_minority_over, y_majority), axis=0)
    
    #shuffle
    X_over, y_over = parallel_permute(X_over, y_over)
    
    return X_over, y_over


def get_top_feat_df(coefs, df, clf_name='classifer'):
    """
    Returns dataframe of features sorted in descending order with feature names as index
    clf_feats(list): feature coefficient weights
    df(pandas): panadas dataframe, target column MUST BE LAST COLUMN
    """
    
    feats_desc_args = np.argsort(coefs)[::-1]
    feats_desc = coefs[feats_desc_args]
    
    df_feat = pd.DataFrame(data=feats_desc, index=df.columns[feats_desc_args], columns=[clf_name])
    
    return df_feat


def pickle_save(model, location, mode='wb'):
    """
    saves an object to a file location
    model = object name
    location = location to dump
    mode = mode in which the file is opened
    
    """
    pickle.dump(model, 
                open(file= location, mode= mode))
    print('model saved to: {}'.format(location))
    

def pickle_load(location, mode='rb'):
    """
    loads an object from a file location
    location = location to read from
    mode = mode in which the file is opened
    
    """
    loaded_model = pickle.load(open(file= location, mode=mode))
    return loaded_model
