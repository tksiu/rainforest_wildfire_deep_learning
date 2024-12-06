import numpy as np


## create time lags by shifting the series/sequence

def create_time_lag(combined_feature, req_time, lags):
    lag_features = []
    for g in combined_feature:
        grid_lag_feature = []
        for t in req_time:
            arr_window = g[(t-lags):t].reshape(1, lags, g.shape[1], g.shape[2], g.shape[3])
            grid_lag_feature.append(arr_window)
        arr_g = np.concatenate(tuple(grid_lag_feature), axis = 0)
        lag_features.append(arr_g)
    return lag_features



## create an adjacency (binary) matrix for the spatial relationship between the local grids
##  method 1)  apply matrix multiplication to extract the corresponding grid's feature arrays --> sparse matrix resuming each grid index
##  method 2)  recall the feature arrays by indexing, with an order of north, east, south, west, plus the focal grid itself

##  Rook contiguity (or four cardinal directions)

class SpatialConcat_AmazonForest():
    """"concatenate sub-grids spatially: Spatial Attention"""
    
    def __init__(self, grid_index, features):
        self.grid_index = grid_index
        self.features = features

        ##  matrix method
        self.adjacency_matrix = np.array([
            [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,1,1,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0],
            [0,1,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0],
            [0,0,0,1,1,0,1,1,0,0,0,0,0,1,0,0,0,0,0,0],
            [0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,1,0,0,0],
            [0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0],
            [0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0],
            [0,0,0,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0],
            [0,0,0,0,0,0,1,0,0,0,0,0,1,1,0,1,1,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,1,0],
            [0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,1,1,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1]
            ])
        
        ##  indexing method
        self.adjacency_indices = [
                     [1,2,-1,-1], [4,3,0,-1], [3,-1,-1,0], [6,12,2,1],
                     [5,6,1,10], [20,7,4,11], [7,13,3,4], [21,16,6,5],
                     [9,10,-1,-1], [-1,11,8,-1], [11,4,-1,8], [-1,5,10,9],
                     [13,14,-1,3], [16,15,12,6], [15,-1,-1,12], [18,-1,14,13],
                     [17,18,13,7], [-1,19,16,21], [19,-1,15,16], [-1,-1,18,17]
        ]
    
    def get_vectors(self, method="matrix"):

        if method == "matrix":

            ## engineer a grid-specific matrix
            spatial_vector = self.adjacency_matrix[self.grid_index]
            spatial_matrix = np.diag(spatial_vector)

            ## matrix mupltiplication, 
            ## if either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.
            spa_atten_features = np.matmul(self.features, spatial_matrix)
            
        elif method == "indices":
            
            ## query per each index representing the four directions
            spatial_vector = self.adjacency_indices[self.grid_index]

            spa_atten_features = []
            for g in [self.grid_index] + spatial_vector:
                if g != -1:
                    spa_atten_features.append(self.features[:,:,:,:,:,g].\
                                              reshape(self.features.shape[0], 
                                                      self.features.shape[1], 
                                                      self.features.shape[2], 
                                                      self.features.shape[3], 
                                                      self.features.shape[4], 1))
                else:
                    spa_atten_features.append(np.zeros((self.features.shape[0], 
                                                        self.features.shape[1], 
                                                        self.features.shape[2], 
                                                        self.features.shape[3], 
                                                        self.features.shape[4], 1)))
                    
            spa_atten_features = np.concatenate(tuple(spa_atten_features), axis = 5)

        else:
            raise ValueError("Invalid method specified, choose one of the followings: 'matrix' or 'indices'.")

        return spa_atten_features

