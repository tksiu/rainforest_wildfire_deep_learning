import numpy as np


def feature_transformation_era5_land(feature_dict: dict):

    feature_dict['temperature'] = [np.power(x, 3) for x in feature_dict['temperature']]
    feature_dict['precipitation'] = [np.log(x) for x in feature_dict['precipitation']]
    feature_dict['pressure'] = [np.power(x, 3) for x in feature_dict['pressure']]
    feature_dict['runoff'] = [np.log(x) for x in feature_dict['runoff']]
    feature_dict['soil_temperature'] = [np.power(x, 3) for x in feature_dict['soil_temperature']]
    for x in feature_dict['soil_water']:
        x[:,:,:,0] = np.log(x[:,:,:,0])
    for x in feature_dict['albedo']:
        x[x > 0.3] = 0.3
    for x in range(len(feature_dict['leaf'])):
        leaf_new = np.sum(feature_dict['leaf'][x][:,:,:,:], axis = 3).reshape(-1,50,50,1)
        feature_dict['leaf'][x] = leaf_new
    for x in range(len(feature_dict['evaporation'])):
        feature_dict['evaporation'][x][:,:,:,1] = np.power(feature_dict['evaporation'][x][:,:,:,1], 3)
        feature_dict['evaporation'][x][:,:,:,2] = np.power(feature_dict['evaporation'][x][:,:,:,2], 3)
        feature_dict['evaporation'][x][:,:,:,4] = np.power(feature_dict['evaporation'][x][:,:,:,4], 3)
        feature_dict['evaporation'][x] = feature_dict['evaporation'][x][:,:,:,np.r_[0:3,4,6]]
    
    return feature_dict

def feature_transformation_era5(feature_dict: dict):

    feature_dict['temperature'] = [np.power(x, 3) for x in feature_dict['temperature']]
    feature_dict['precipitation'] = [np.log(x) for x in feature_dict['precipitation']]
    feature_dict['pressure'] = [np.power(x, 3) for x in feature_dict['pressure']]
    
    return feature_dict


def get_global_min_max(feature_list: list):
    feature_max = np.max(np.concatenate([y.reshape(1, -1) for y in [np.max(x.reshape(x.shape[0] * x.shape[1], -1), axis=0) for x in feature_list]]), axis=0)
    feature_min = np.min(np.concatenate([y.reshape(1, -1) for y in [np.min(x.reshape(x.shape[0] * x.shape[1], -1), axis=0) for x in feature_list]]), axis=0)
    return feature_max, feature_min


def calculate_min_max(feature_list: list, feature_max: float, feature_min: float):
    scaled = [[(y - feature_min) / (feature_max - feature_min) for y in x] for x in feature_list]
    scaled = [[np.nan_to_num(y, nan=np.nanmean(y)) for y in x] for x in feature_list]
    return scaled


