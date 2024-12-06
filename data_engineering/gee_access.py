import numpy as np
import ee
import time


####  1)  splitting the window (list) into equal-sized grids

def grid_split(grid_list: list, split_size: int):

    land_grid = []
    land_point = []
    land_grid_label = []
    
    for grid in grid_list:
        grid_local = []
        point_local = []
        for gx in range(0, int(grid[1] - grid[0])):
            for gy in range(0, int(grid[3] - grid[2])):
                lat_low = grid[2] + gy
                lat_high = grid[2] + gy + 1
                lon_low = grid[0] + gx
                lon_high = grid[0] + gx + 1
                grid_label = str(lat_low) + "-" + str(lat_high) + "-" + str(lon_low) + "-" + str(lon_high) + "-"
                land_grid_label.append(grid_label)
                
                temp_per_grid = []
                temp_central_point_per_grid = []
                for x in np.arange(0, 1, split_size):
                    for y in np.arange(0, 1, split_size):
                        temp_per_grid.append((grid[0] + gx + x, 
                                              grid[0] + gx + x + split_size,
                                              grid[2] + gy + y,
                                              grid[2] + gy + y + split_size))
                        poi_y = round(grid[2] + gy + y + split_size/2, 2)
                        poi_x = round(grid[0] + gx + x + split_size/2, 2)
                        temp_central_point_per_grid.append([poi_x, poi_y])
                
                grid_local.append(temp_per_grid)
                point_local.append(temp_central_point_per_grid)
        
        land_grid.append(grid_local)
        land_point.append(point_local)
        
    return land_grid, land_point, land_grid_label




###  2)  downloading GEE images

def batch_download_image(grid, start_date, end_date, 
                         image_collection, feature_layers, scale,
                         output_folder, visual_settings_max, visual_settings_min, visual_settings_palette = None):
    
    ## specify collection
    collection = ee.ImageCollection(image_collection).select(feature_layers).filterDate(start_date, end_date)
    
    ## specify image color scale/range
    def ImageVis(img, min, max, palette=None):
        vis = img.visualize(
                min = min,
                max = max,
                palette = palette
            )
        return vis

    vis = map(lambda i: ImageVis(i, visual_settings_max, visual_settings_min, visual_settings_palette), collection)

    ## exporting
    for g in range(len(grid)):

        boundary_filter = [grid[g][0], grid[g][2], grid[g][1], grid[g][3]]
        region = ee.Geometry.Rectangle(boundary_filter)
        
        ##     @ before 2024:  geetools
        """
        tasks = geetools.batch.Export.imagecollection.toDrive(
                  collection = vis, 
                  folder = output_folder + '-' + 'grid_index_' + str(g),
                  scale = scale,
                  region = region,
                  crs = 'EPSG:4326',
                  fileFormat = 'GeoTIFF')
        """
        ##     @ after 2024:  geetools deprecated batch export for image collections

        img_list = vis.toList(collection.size())

        counter = 0

        while counter >= 0:

            try:
                img = ee.Image(img_list.get(counter))
                date = ee.Date(img.get('system:time_start')).format(None, 'UTC').getInfo()
            except:
                print("End.")
                break

            task = ee.batch.Export.image.toDrive(**{
                    'image': img,
                    'description': 'TROPOMI_NO2_' + date,
                    'folder': output_folder + '-' + 'grid_index_' + str(g),
                    'region': region,
                    'scale': scale,
                    'crs':'EPSG:4326',
                    'fileFormat':'GeoTIFF',
                })
            task.start()

            counter += 1
            
            time.sleep(0.1) 

        print("Completed series: " + str(g+1))




###  3)  downloading GEE arrays as tables

def batch_download_csv(grid_points, grid_labels, start_date, end_date, 
                feature_collection, feature_layers, scale,
                output_folder, reduce_region = False, spatial_pool_method = None, year_label = ""):

    ## specify collection
    collection = ee.ImageCollection(feature_collection).select(feature_layers).filterDate(start_date, end_date)
    
    ## export mapping function
    def create_export(values):
        return ee.Feature(None, ee.Dictionary.fromLists(feature_names, ee.List(values)))

    ## reduceRegion pooling
    def local_pooling(image, pool_method="mean"):
        if pool_method == "mean":
            reduced_dict = image.reduceRegion(
                                reducer = ee.Reducer.mean(),
                                geometry = multipoints,
                                scale = scale)
        elif pool_method == "max":
            reduced_dict = image.reduceRegion(
                                reducer = ee.Reducer.max(),
                                geometry = multipoints,
                                scale = scale)
        elif pool_method == "min":
            reduced_dict = image.reduceRegion(
                                reducer = ee.Reducer.min(),
                                geometry = multipoints,
                                scale = scale)
        elif pool_method == "median":
            reduced_dict = image.reduceRegion(
                                reducer = ee.Reducer.median(),
                                geometry = multipoints,
                                scale = scale)
        elif pool_method == "sum":
            reduced_dict = image.reduceRegion(
                                reducer = ee.Reducer.sum(),
                                geometry = multipoints,
                                scale = scale)
        else:
            raise ValueError("Pooling method must be in ('mean','max','min','median','sum').")
        return image.set(reduced_dict)

    for cp in range(len(grid_points)):

        ## list of points in the grid
        multipoints = ee.Geometry.MultiPoint(grid_points[cp])
        ## get data for each grid point
        get_collection = collection.filterBounds(multipoints).select(feature_collection)
        ## spatial pooling if needed
        if reduce_region == True:
            get_collection = map(lambda i: local_pooling(i, pool_method=spatial_pool_method), get_collection)
        ## processing
        get_collection_array = get_collection.getRegion(multipoints, scale)
        feature_names = get_collection_array.get(0)
        get_export = ee.FeatureCollection(get_collection_array.map(create_export))

        ## exporting
        complete = 0
        tasks = ee.batch.Export.table.toDrive(**{
            'collection': get_export,
            'description': 'table_' + str(grid_labels[cp]) + "_" + year_label,
            'folder': output_folder,
            'fileFormat':'CSV',
            'selectors': ['id', 'longitude', 'latitude', 'time'] + feature_collection
        })
        tasks.start()

        while complete == 0:
            if tasks.status()['state'] == 'COMPLETED':
                complete = 1
            else:
                complete = 0
        if (cp+1) % 20 == 0:
            print("Completed series: " + str(cp+1))

        time.sleep(0.1) 



