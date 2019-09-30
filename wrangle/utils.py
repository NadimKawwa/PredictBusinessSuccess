import os
import glob
import pandas as pd
import numpy as np


def chunk_loader(directory, c_size=100_000, orient='columns', lines=True, read_limit=0):
    """
    Reads a directory in chunks, infers type if json or csv format
    directory(str) = location of file
    c_size(int) = size of individual chunk
    oritnet(str) = orientation of expected format
    read_limit(int) = limit of file to read
    """
    
    if '.json' in directory:
        #return JsonReader object
        review_df_chunk = pd.read_json(directory, orient=orient,lines=lines, chunksize=c_size)
    elif '.csv' in directory:
        review_df_chunk = pd.read_csv(directory, chunksize=c_size)
    else:
        print('unsopported file type, must be jsson or csv')
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
        
     