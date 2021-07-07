
#########
# ANNOY #
#########
# https://github.com/spotify/annoy

from annoy import AnnoyIndex
import random
import pandas as pd
import time
import os
import numpy as np

def create_random_data( nrow, ncol ):
    df = pd.DataFrame( [ 
                            [ random.uniform(-20,100) for col in range(ncol) ]
                            for row in range(nrow)    
                       ]
                    )
    df.columns = [ f'x{colnum+1}' for colnum in range(ncol) ]       # name the columns x1,x2,...
    return df

features_X = create_random_data( nrow=1000000, ncol=100 )

# example of how ANNOY algorithm works -------------------------------------------------------------------------------------------
#f = features_X.shape[1]           # define vector dimension (i.e. number of features)   
#t = AnnoyIndex(f, 'euclidean')    # create annoy index, defining distance metric and vector dimension 
#                                  # available metrics are "angular", "euclidean", "manhattan", "hamming", or "dot"
#for i in range(len(features_X)):  # add items to annoy index (i.e. add rows of feature matrix)
#    t.add_item(i, features_X.iloc[i].to_numpy() )     # i is the index assigned to each item
#
#t.get_n_items()          # see number of items in the index       
#
## build the trees
#t.build( n_trees = 10,         # number of trees to build (more trees = more accurate results & longer build time & larger indexes)
#                               # set n_trees as large as possible, given the amount of memory you can afford  
#          n_jobs = -1          # number of threads to use (use -1 for all)
#       ) 
#dir(t)  # see structure of annoy object
#
## you can save the built model like this:
#t.save('test.ann')         # save the ANN object to disk
#
## the object is loaded like this:
#u = AnnoyIndex(f, 'euclidean')          
#u.load('test.ann')   # super fast, will just mmap the file
#u.get_nns_by_item( i = 0,                       # index of item (row) to find neighbours for
#                   n = 3,                       # number of neighbours to fetch 
#                   search_k = -1,               # number of nodes to search. large search_k = more accurate & slower to run (default = n_trees*n)
#                                                # set as large as possible given your time constraints 
#                   include_distances = False 
#                 )

# run speed tests:
os.chdir('C://Users//jbolton//Documents//blogs//')       # set working directory  
file = open('annoy_speed_tests.txt', 'w')                # create a new text file if it doesn't exist 
file.write('test_number|n_trees|search_k|build_time|min_searchtime|searchtime_0.25|median_searchtime|searchtime_0.75|max_searchtime|min_dist|dist_0.25|median_dist|dist_0.75|max_dist')

tests_to_run = [(100,100), (100,1000), (1000,100), (1000,1000)]          # each tuple (a,b) is a test of annoy, with parameters n_trees=a & search_k=b    

# when assessing/comparing different nearest-neighbour algorithms, we are going to find this many neighbours for this many items: 
n_items_to_find_neighbours_for = 100                        # we're going to get (random) neighbours for this many items 
n_neighbours_to_find_per_item = 20                          # this is how many (random) neighbours we'll pull for each item

# first, set benchmark simply choosing random neighbours:
result_string = 'random_neighbours||||||||'                 # number this test as "random_neighbours", and make other irrelevant results blank
store_item_distances = []                                   # list in which to store the euclidean distances 
for item_i in range(n_items_to_find_neighbours_for):
    item_i_vec = features_X.iloc[ random.randint(0, len(features_X) ) ].to_numpy()           # pick a random item ("item_i") to find neighbours for
    for nbor_j in range(n_neighbours_to_find_per_item ):                                     # find random neighbours for item_i
        nbor_j_vec = features_X.iloc[ random.randint(0, len(features_X) ) ].to_numpy()       # extract a random neighbour  
        store_item_distances = store_item_distances + [np.linalg.norm(item_i_vec-nbor_j_vec)]  # store euclidean distance between item_i and nbor_j
# save summary of distances to found (random) neighbours to result_string: 
result_string = ( result_string + 
                  '|' +
                  str(min(store_item_distances)) +
                  '|' + 
                  str(np.quantile(store_item_distances,0.25)) +
                  '|' + 
                  str(np.quantile(store_item_distances,0.5)) +
                  '|' +
                  str(np.quantile(store_item_distances,0.75)) +
                  '|' +
                  str(max(store_item_distances))
                )
file.write( '\n'+result_string )     # write the random neighbours results to the test file

# save results for a second (this time unreachable) benchmark: where each item gets their actual closest neighbours (impossible in practice):
result_string = 'perfect_neighbours|||'                 # number this test as "perfect_neighbours", and make other irrelevant results blank
store_search_times = []
store_item_distances = []
for item_i in range(n_items_to_find_neighbours_for):
    item_i_vec = features_X.iloc[ random.randint(0, len(features_X) ) ].to_numpy()           # pick a random item ("item_i") to find neighbours for
    start_time = time.perf_counter()                                                         # used to time how long the exhaustive search takes 
    # find all closest neighbours for item_i
    distances_of_item_i_to_all_other_items = features_X.apply( lambda row: np.linalg.norm( item_i_vec - row.to_numpy() ), axis=1 )
    item_i_closest_neighbour_distances = distances_of_item_i_to_all_other_items.sort_values(ascending=True).iloc[1:n_neighbours_to_find_per_item]
    end_time = time.perf_counter()
    store_search_times.append( end_time-start_time )                                      # store the time (in seconds) taken to find neighbours for this item                                  
    store_item_distances = store_item_distances + list(item_i_closest_neighbour_distances) # store euclidean distance between item_i and each of it's nearest neighbours
# save summary of times to find (maximally close) neighbours to result_string: 
result_string = ( result_string + 
                  '|' +
                  str(min(store_search_times)) +
                  '|' + 
                  str(np.quantile(store_search_times,0.25)) +
                  '|' + 
                  str(np.quantile(store_search_times,0.5)) +
                  '|' +
                  str(np.quantile(store_search_times,0.75)) +
                  '|' +
                  str(max(store_search_times))
                )
# save summary of distances to found (maximally close) neighbours to result_string: 
result_string = ( result_string + 
                  '|' +
                  str(min(store_item_distances)) +
                  '|' + 
                  str(np.quantile(store_item_distances,0.25)) +
                  '|' + 
                  str(np.quantile(store_item_distances,0.5)) +
                  '|' +
                  str(np.quantile(store_item_distances,0.75)) +
                  '|' +
                  str(max(store_item_distances))
                )
file.write( '\n'+result_string )     # write the perfect neighbours results to the test file

# run the tests of annoy, with different parameters:
for test_i in range(len(tests_to_run)):       # iterate through all of the tests 
    result_string = ''
    n_trees_this_test = tests_to_run[test_i][0]
    search_k_this_test = tests_to_run[test_i][1]
    result_string = result_string + 'annoy_' + str(test_i+1) + '|'                     # save test number to result string
    result_string = result_string + str(n_trees_this_test) + '|'                           # save n_trees parameter to result string 
    result_string = result_string + str(search_k_this_test) + '|'    # save search_k parameter to result string

    f = features_X.shape[1]           # define vector dimension (i.e. number of features)       
    t = AnnoyIndex(f, 'euclidean')    # create annoy index
    for i in range(len(features_X)):  # add items to annoy index (i.e. add rows of feature matrix)
        t.add_item(i, features_X.iloc[i].to_numpy() )     # i is the index assigned to each item

    # time the build:    
    start_time = time.perf_counter() 
    t.build( n_trees = n_trees_this_test,         
              n_jobs = -1          # use all available threads
           ) 
    end_time = time.perf_counter()
    result_string = result_string + str( (end_time-start_time)/60 )            # save build time to result_string

    # time nearest-neighbour searches for several random items:
    store_search_times = []
    store_item_distances = []
    for item_i in range(n_items_to_find_neighbours_for):
        # time the search for neighbours for this item:
        start_time = time.perf_counter() 
        get_neighbours =  t.get_nns_by_item( i = random.randint(0, len(features_X) ),     # a random item 
                                             n = (n_neighbours_to_find_per_item+1),       # number of neighbours to fetch for this item 
                                                                                          # find an extra neighbour since the 1st closest neighbour is the point itself
                                             search_k = search_k_this_test,               # number of nodes to search. large search_k = more accurate & slower to run (default = n_trees*n)
                                             include_distances = True 
                                           )
        end_time = time.perf_counter()
        store_search_times.append( end_time-start_time )                        # store the time taken to find neighbours for this item                                  
        store_item_distances = store_item_distances + get_neighbours[1][1:]     # store all of the distances between the item and it's found neighbours (excluding the point itself)     
    # save summary of search for neighbour times to result_string: 
    result_string = ( result_string + 
                      '|' +
                      str(min(store_search_times)) + 
                      '|' + 
                      str(np.quantile(store_search_times,0.25)) +
                      '|' + 
                      str(np.quantile(store_search_times,0.5)) +
                      '|' + 
                      str(np.quantile(store_search_times,0.75)) +
                      '|' + 
                      str(max(store_search_times))
                    )
    # save summary of distances to found neighbours per result_string: 
    result_string = ( result_string + 
                      '|' +
                      str(min(store_item_distances)) +
                      '|' + 
                      str(np.quantile(store_item_distances,0.25)) +
                      '|' + 
                      str(np.quantile(store_item_distances,0.5)) +
                      '|' +
                      str(np.quantile(store_item_distances,0.75)) +
                      '|' +
                      str(max(store_item_distances))
                    )

    file.write( '\n'+result_string )                                          # write the result string to file

file.close()

