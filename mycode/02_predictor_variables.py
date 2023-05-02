import rasterio
import geopandas as gpd
import pandas as pd
import numpy as np
from constants import *
import os

# Set-up working environment

# Extract predictor variables_fig1 and align the raster cells

def clip_rasters():
    # Open raster

    # Clip raster to study site

    # Align raster cells

    # Save output
    pass

# Extract predictor variables_fig1 to dataset without performing bilinear interpolation
def extract_predictor_variables(gdf, variable_dict, output):
    assert output.endswith('.csv')
    for variable in variable_dict.keys():
        print(f'Extracting data of {variable}')
        if os.path.exists(variable_dict[variable]):
            src = rasterio.open(variable_dict[variable])
            if src.crs != gdf.crs:
                gdf.to_crs(src.crs, inplace=True)
            no_data = src.nodata
            coord_list = [(x, y) for x, y in zip(gdf['geometry'].x, gdf['geometry'].y)]
            gdf[variable] = [x[0] if x[0] != no_data else np.nan for x in src.sample(coord_list) ]
        else:
            print(f'No file for {variable}')
    gdf.to_csv(output, index=False)


df = pd.read_csv(dataset)
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x=df.longitude, y=df.latitude), crs=crs)


# all_features = features + river_features
all_features = features

for feature in all_features:
    if feature in df.columns:
        df.drop(columns=feature, inplace=True)

all_features_dict = {feature: os.path.join(file_dir, feature+'.tif') for feature in all_features}

extract_predictor_variables(gdf=gdf, variable_dict=all_features_dict,
                            output=os.path.join(dataset))
