
#########
# ANNOY #
#########
# https://github.com/spotify/annoy

from annoy import AnnoyIndex
import random
import pandas as pd
import time
import os

# here is how to time number of seconds that code takes to run:
start_time = time.perf_counter() 
end_time = time.perf_counter()
end_time - start_time                   # number of seconds between start_time and end_time

def create_random_data( nrow, ncol ):
    df = pd.DataFrame( [ 
                            [ random.uniform(-20,100) for col in range(ncol) ]
                            for row in range(nrow)    
                       ]
                    )
    df.columns = [ f'x{colnum+1}' for colnum in range(ncol) ]       # name the columns x1,x2,...
    return df

features_X = create_random_data( nrow=8, ncol=4 )

f = features_X.shape[1]           # define vector dimension (i.e. number of features)   
t = AnnoyIndex(f, 'euclidean')    # create annoy index, defining distance metric and vector dimension 
                                  # available metrics are "angular", "euclidean", "manhattan", "hamming", or "dot"
for i in range(len(features_X)):  # add items to annoy index (i.e. add rows of feature matrix)
    t.add_item(i, features_X.iloc[i].to_numpy() )     # i is the index assigned to each item

t.get_n_items()          # see number of items in the index       

# build the trees
t.build( n_trees = 10,         # number of trees to build (more trees = more accurate results & longer build time & larger indexes)
                               # set n_trees as large as possible, given the amount of memory you can afford  
           njobs = -1          # number of threads to use (use -1 for all)
       ) 
dir(t)  # see structure of annoy object

# you can save the built model like this:
t.save('test.ann')         # save the ANN object to disk

# the object is loaded like this:
u = AnnoyIndex(f, 'euclidean')          
u.load('test.ann')   # super fast, will just mmap the file
u.get_nns_by_item( i = 0,                       # index of item (row) to find neighbours for
                   n = 3,                       # number of neighbours to fetch 
                   search_k = -1,               # number of nodes to search. large search_k = more accurate & slower to run (default = n_trees*n)
                                                # set as large as possible given your time constraints 
                   include_distances = False 
                 )

# run speed tests:
os.chdir('C://Users//jbolton//Documents//blogs//')       # set working directory  
file = open('annoy_speed_tests.txt', 'w')                # create a new text file if it doesn't exist 
file.write('test_number|n_trees|search_k|min_time|lwr_qtl_time|median_time|upr_qtl_time|max_time|min_distance|lwr_qtl_distance|median_distance|upr_qtl_distance|max_distance')
file.close()

tests_to_run = [(10,10), (10,100), (100,10), (100,100)]          # each tuple (a,b) is a test of annoy, with n_trees=a & search_k=b    

for test_i in range(len(tests_to_run)):       # iterate through all of the tests 
    result_string = ''
    n_trees_this_test = tests_to_run[test_i][0]
    search_k_this_test = tests_to_run[test_i][1]
    result_string = result_string + str(test_i+1) + '|'                     # save test number to result string
    result_string = result_string + str( n_trees_this_test ) + '|'                           # save n_trees parameter to result string 
    result_string = result_string + str( search_k_this_test ) + '|'    # save search_k parameter to result string

    t = AnnoyIndex(f, 'euclidean')    # create annoy index
    for i in range(len(features_X)):  # add items to annoy index (i.e. add rows of feature matrix)
        t.add_item(i, features_X.iloc[i].to_numpy() )     # i is the index assigned to each item

    # time the build:    
    start_time = time.perf_counter() 
    t.build( n_trees = 10,         # number of trees to build (more trees = more accurate results & longer build time & larger indexes)
                               # set n_trees as large as possible, given the amount of memory you can afford  
               njobs = -1          # number of threads to use (use -1 for all)
       ) 
    end_time = time.perf_counter()
    end_time - start_time                   # number of seconds between start_time and end_time
