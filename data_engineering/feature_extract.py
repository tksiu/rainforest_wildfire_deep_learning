import cv2
import pandas as pd
import numpy as np
from datetime import datetime


## function for burned area images
def burned_area_extract_func(original_dim, image_folder, image_series, 
                             pool = False, reduced_dim = None):

    max_frp = []
    for n in range(len(image_folder)):
        sub_max_frp = []
        for m in range(len(image_series[n])):
            ## read image and get arrays
            cv_burn = cv2.imread(image_series[n][m], -1)
            cv_burn_resize = cv2.resize(cv_burn, (original_dim, original_dim), interpolation = cv2.INTER_AREA)
            ## color gradients by fire radiative power
            cv_frp = cv_burn_resize[:,:,0]
            cv_frp = np.nan_to_num(cv_frp, nan=0)
            sub_max_frp.append(cv_frp)
        max_frp.append(sub_max_frp)
    
    ## maximum pooling to lower dimension if necessary (for computation resource issues)
    if pool == True:
        if reduced_dim != None:
            if isinstance(reduced_dim, int) == True:

                max_frp_pool = []
                for g in range(len(image_folder)):
                    grid_frp_pool = []
                    for t in range(len(image_series[n])):
                        init_arr_frp = np.zeros((reduced_dim, reduced_dim))
                        for gi in range(reduced_dim):
                            for gj in range(reduced_dim):
                                init_arr_frp[gi,gj] = np.max(
                                    max_frp[g][t][int(gi*original_dim/reduced_dim):int(gi*original_dim/reduced_dim + original_dim/reduced_dim), 
                                                  int(gj*original_dim/reduced_dim):int(gj*original_dim/reduced_dim + original_dim/reduced_dim)]
                                )
                        grid_frp_pool.append(init_arr_frp)
                    max_frp_pool.append(grid_frp_pool)
            else:
                raise ValueError("reduced dimension must be integer.")
        else:
            raise ValueError("None value provided.")

    return max_frp_pool


## function for ERA5 weather features
def era5_weather_extract_func(csv_path, csv_name, window_width, window_height):

    records = list()

    for i in range(len(csv_name)):

        temp_path = csv_path + csv_name[i]
        temp_csv = pd.read_csv(temp_path)
        temp_csv = temp_csv.iloc[1:,:]
        temp_csv['time'] = temp_csv['time'].apply(lambda x: datetime.fromtimestamp(int(float(x))/1000))

        iter = 0
        local_list = []

        for t in range(len(sorted(list(set(temp_csv['time']))))):

            grid_list = {}
            local_arr_temperature = np.zeros((4,4,4))
            local_arr_precipitation = np.zeros((4,4,1))
            local_arr_pressure = np.zeros((4,4,2))
            local_arr_wind = np.zeros((4,4,2))

            for px in range(4):
                for py in range(4):

                    local_arr_temperature[4-1-px, py] = np.array(temp_csv.iloc[iter,4:8]).astype('float')
                    local_arr_precipitation[4-1-px, py] = np.array(temp_csv.iloc[iter,8]).astype('float')
                    local_arr_pressure[4-1-px, py] = np.array(temp_csv.iloc[iter,9:11]).astype('float')
                    local_arr_wind[4-1-px, py] = np.array(temp_csv.iloc[iter,11:13]).astype('float')
                    iter += 1

            grid_list['temperature'] = local_arr_temperature
            grid_list['precipitation'] = local_arr_precipitation
            grid_list['pressure'] = local_arr_pressure
            grid_list['wind'] = local_arr_wind

            local_list.append(grid_list)

        records.append(local_list)

    ts_bands = {}
    ts_bands['temperature'] = []
    ts_bands['precipitation'] = []
    ts_bands['pressure'] = []
    ts_bands['wind'] = []

    for t in range(len(sorted(list(set(temp_csv['time']))))):

        global_arr_temperature = np.zeros((window_height*4,window_width*4,4))
        global_arr_precipitation = np.zeros((window_height*4,window_width*4,1))
        global_arr_pressure = np.zeros((window_height*4,window_width*4,2))
        global_arr_wind = np.zeros((window_height*4,window_width*4,2))

        iter = 0

        for j in range(0, window_width*4, 10):
            for i in range(0, window_height*4, 10):
                global_arr_temperature[window_height*4-(i+4):(window_height*4-i),j:(j+4),:] = records[iter][t]['temperature']
                global_arr_precipitation[window_height*4-(i+4):(window_height*4-i),j:(j+4),:] = records[iter][t]['precipitation']
                global_arr_pressure[window_height*4-(i+4):(window_height*4-i),j:(j+4),:] = records[iter][t]['pressure']
                global_arr_wind[window_height*4-(i+4):(window_height*4-i),j:(j+4),:] = records[iter][t]['wind']
                iter += 1

        ts_bands['temperature'].append(global_arr_temperature)
        ts_bands['precipitation'].append(global_arr_precipitation)
        ts_bands['pressure'].append(global_arr_pressure)
        ts_bands['wind'].append(global_arr_wind)
      
    return ts_bands



## function for ERA5 land features
def era5_land_extract_func(csv_path, csv_subpath, csv_name, grids, grid_dim, frequency="monthly", hourly_data_aggregation=None, num_fire_seasons=None):

    if frequency not in ['monthly', 'hourly']:
        raise ValueError("Data frequency must be either 'monthly' or 'hourly'.")
    else:

        records = list()
        count = 0

        for k in range(len(csv_subpath)):

            sublist = []

            if frequency == "monthly":
                stop_criteria = count + grid_dim ** 2
            elif frequency == "hourly":
                stop_criteria = count + num_fire_seasons * grid_dim ** 2

            i = count
            while i < stop_criteria:

                temp_path = csv_path + csv_subpath[k] + '/' + csv_name[i]
                temp_csv = pd.read_csv(temp_path)
                temp_csv = temp_csv.iloc[1:,:]

                if frequency == "monthly":
                    temp_csv['time'] = temp_csv['time'].apply(lambda x: datetime.fromtimestamp(int(float(x))/1000))
                elif frequency == "hourly":
                    temp_csv['time'] = temp_csv['time'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").date())
                    if hourly_data_aggregation == "mean":
                        temp_csv = temp_csv.groupby(['time']).mean()
                    elif hourly_data_aggregation == "max":
                        temp_csv = temp_csv.groupby(['time']).max()
                    elif hourly_data_aggregation == "min":
                        temp_csv = temp_csv.groupby(['time']).min()
                    elif hourly_data_aggregation == "sum":
                        temp_csv = temp_csv.groupby(['time']).sum()
                    elif hourly_data_aggregation == "median":
                        temp_csv = temp_csv.groupby(['time']).median()
                    else:
                        raise ValueError("Aggregation must be in ('mean','max','min','median','sum')")
                    temp_csv = temp_csv.reset_index()

                iter = 0
                local_list = []

                for t in range(len(sorted(list(set(temp_csv['time']))))):

                    grid_list = {}
                    local_arr_temperature = np.zeros((10,10,2))
                    local_arr_pressure = np.zeros((10,10,1))
                    local_arr_precipitation = np.zeros((10,10,1))
                    local_arr_wind = np.zeros((10,10,2))
                    local_arr_soil_temperature = np.zeros((10,10,5))
                    local_arr_soil_water = np.zeros((10,10,5))
                    local_arr_lake = np.zeros((10,10,7))
                    local_arr_snow = np.zeros((10,10,8))
                    local_arr_albedo = np.zeros((10,10,1))
                    local_arr_radiation = np.zeros((10,10,6))
                    local_arr_evaporation = np.zeros((10,10,7))
                    local_arr_runoff = np.zeros((10,10,3))
                    local_arr_leaf = np.zeros((10,10,2))

                    for px in range(10):
                        for py in range(10):

                            local_arr_temperature[10-1-px, py] = np.array(temp_csv.iloc[iter,4:6]).astype('float')
                            local_arr_pressure[10-1-px, py] = np.array(temp_csv.iloc[iter,8]).astype('float')
                            local_arr_precipitation[10-1-px, py] = np.array(temp_csv.iloc[iter,9]).astype('float')
                            local_arr_wind[10-1-px, py] = np.array(temp_csv.iloc[iter,6:8]).astype('float')
                            local_arr_soil_temperature[10-1-px, py] = np.array(temp_csv.iloc[iter,10:15]).astype('float')
                            local_arr_soil_water[10-1-px, py] = np.array(temp_csv.iloc[iter,15:20]).astype('float')
                            local_arr_lake[10-1-px, py] = np.array(temp_csv.iloc[iter,20:27]).astype('float')
                            local_arr_snow[10-1-px, py] = np.array(temp_csv.iloc[iter,27:35]).astype('float')
                            local_arr_albedo[10-1-px, py] = np.array(temp_csv.iloc[iter,35]).astype('float')
                            local_arr_radiation[10-1-px, py] = np.array(temp_csv.iloc[iter,36:42]).astype('float')
                            local_arr_evaporation[10-1-px, py] = np.array(temp_csv.iloc[iter,42:49]).astype('float')
                            local_arr_runoff[10-1-px, py] = np.array(temp_csv.iloc[iter,49:52]).astype('float')
                            local_arr_leaf[10-1-px, py] = np.array(temp_csv.iloc[iter,52:54]).astype('float')
                            iter += 1

                    grid_list['temperature'] = local_arr_temperature
                    grid_list['precipitation'] = local_arr_precipitation
                    grid_list['pressure'] = local_arr_pressure
                    grid_list['wind'] = local_arr_wind
                    grid_list['soil_temperature'] = local_arr_soil_temperature
                    grid_list['soil_water'] = local_arr_soil_water
                    grid_list['lake'] = local_arr_lake
                    grid_list['snow'] = local_arr_snow
                    grid_list['albedo'] = local_arr_albedo
                    grid_list['radiation'] = local_arr_radiation
                    grid_list['evaporation'] = local_arr_evaporation
                    grid_list['runoff'] = local_arr_runoff
                    grid_list['leaf'] = local_arr_leaf

                    local_list.append(grid_list)

                sublist.append(local_list)

            if frequency == "monthly":
                count += grid_dim ** 2
            elif frequency == "hourly":
                count += num_fire_seasons * grid_dim ** 2

            records.append(sublist)

        bands = []
        for g in range(len(grids)):

            ts_bands = {}
            ts_bands['temperature'] = []
            ts_bands['precipitation'] = []
            ts_bands['pressure'] = []
            ts_bands['wind'] = []
            ts_bands['soil_temperature'] = []
            ts_bands['soil_water'] = []
            ts_bands['lake'] = []
            ts_bands['snow'] = []
            ts_bands['albedo'] = []
            ts_bands['radiation'] = []
            ts_bands['evaporation'] = []
            ts_bands['runoff'] = []
            ts_bands['leaf'] = []

            for t in range(len(sorted(list(set(temp_csv['time']))))):
                global_arr_temperature = np.zeros((grid_dim*10,grid_dim*10,2))
                global_arr_precipitation = np.zeros((grid_dim*10,grid_dim*10,1))
                global_arr_pressure = np.zeros((grid_dim*10,grid_dim*10,1))
                global_arr_wind = np.zeros((grid_dim*10,grid_dim*10,2))
                global_arr_soil_temperature = np.zeros((grid_dim*10,grid_dim*10,5))
                global_arr_soil_water = np.zeros((grid_dim*10,grid_dim*10,5))
                global_arr_lake = np.zeros((grid_dim*10,grid_dim*10,7))
                global_arr_snow = np.zeros((grid_dim*10,grid_dim*10,8))
                global_arr_albedo = np.zeros((grid_dim*10,grid_dim*10,1))
                global_arr_radiation = np.zeros((grid_dim*10,grid_dim*10,6))
                global_arr_evaporation = np.zeros((grid_dim*10,grid_dim*10,7))
                global_arr_runoff = np.zeros((grid_dim*10,grid_dim*10,3))
                global_arr_leaf = np.zeros((grid_dim*10,grid_dim*10,2))

                iter = 0
                for j in range(0, 5*10, 10):
                    for i in range(0, 5*10, 10):
                        global_arr_temperature[grid_dim*10-(i+10):(grid_dim*10-i),j:(j+10),:] = records[g][iter][t]['temperature']
                        global_arr_precipitation[grid_dim*10-(i+10):(grid_dim*10-i),j:(j+10),:] = records[g][iter][t]['precipitation']
                        global_arr_pressure[grid_dim*10-(i+10):(grid_dim*10-i),j:(j+10),:] = records[g][iter][t]['pressure']
                        global_arr_wind[grid_dim*10-(i+10):(grid_dim*10-i),j:(j+10),:] = records[g][iter][t]['wind']
                        global_arr_soil_temperature[grid_dim*10-(i+10):(grid_dim*10-i),j:(j+10),:] = records[g][iter][t]['soil_temperature']
                        global_arr_soil_water[grid_dim*10-(i+10):(grid_dim*10-i),j:(j+10),:] = records[g][iter][t]['soil_water']
                        global_arr_lake[grid_dim*10-(i+10):(grid_dim*10-i),j:(j+10),:] = records[g][iter][t]['lake']
                        global_arr_snow[grid_dim*10-(i+10):(grid_dim*10-i),j:(j+10),:] = records[g][iter][t]['snow']
                        global_arr_albedo[grid_dim*10-(i+10):(grid_dim*10-i),j:(j+10),:] = records[g][iter][t]['albedo']
                        global_arr_radiation[grid_dim*10-(i+10):(grid_dim*10-i),j:(j+10),:] = records[g][iter][t]['radiation']
                        global_arr_evaporation[grid_dim*10-(i+10):(grid_dim*10-i),j:(j+10),:] = records[g][iter][t]['evaporation']
                        global_arr_runoff[grid_dim*10-(i+10):(grid_dim*10-i),j:(j+10),:] = records[g][iter][t]['runoff']
                        global_arr_leaf[grid_dim*10-(i+10):(grid_dim*10-i),j:(j+10),:] = records[g][iter][t]['leaf']
                        iter += 1

                ts_bands['temperature'].append(global_arr_temperature)
                ts_bands['precipitation'].append(global_arr_precipitation)
                ts_bands['pressure'].append(global_arr_pressure)
                ts_bands['wind'].append(global_arr_wind)
                ts_bands['soil_temperature'].append(global_arr_soil_temperature)
                ts_bands['soil_water'].append(global_arr_soil_water)
                ts_bands['lake'].append(global_arr_lake)
                ts_bands['snow'].append(global_arr_snow)
                ts_bands['albedo'].append(global_arr_albedo)
                ts_bands['radiation'].append(global_arr_radiation)
                ts_bands['evaporation'].append(global_arr_evaporation)
                ts_bands['runoff'].append(global_arr_runoff)
                ts_bands['leaf'].append(global_arr_leaf)

            bands.append(ts_bands)

    return bands
