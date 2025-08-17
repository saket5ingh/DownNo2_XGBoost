import pandas as pd
import numpy as np
from sklearn import preprocessing
import sklearn.metrics as metrics
import pickle
import os
import argparse
import textwrap
from xgboost import XGBRegressor


def get_bounding_box(region_of_interest):
    if region_of_interest == 'ROI1':
        bbox = [6, 12, 42, 48]
    else:
        bbox = [2, 8, 48, 54]        
    return bbox    


def get_mask(dataframe, bbox):
    return ((dataframe.Longitude > bbox[0]) & 
            (dataframe.Longitude < bbox[1]) & 
            (dataframe.Latitude > bbox[2]) & 
            (dataframe.Latitude < bbox[3]))


def make_dummy_all(dataframe):
    filename = os.path.join(root_directory, 'input', 'LUD', 'clc_legend.csv')
    meta_land_use = pd.read_csv(filename, index_col=False, sep=';')
    possibilities = meta_land_use.CLC_CODE.unique()
    exists = dataframe.lu.unique()
    difference = pd.Series([item for item in possibilities if item not in exists])
    target = dataframe.lu.append(pd.Series(difference))
    target = target.reset_index(drop=True)
    dummy = pd.get_dummies(target)
    dummy = dummy.drop(dummy.index[list(range(len(dummy) - len(difference), len(dummy)))])
    target_major = np.floor(target / 100.)
    dummy_major = pd.get_dummies(target_major, prefix='lu_major')
    dummy_major = dummy_major.drop(dummy_major.index[list(range(len(dummy) - len(difference), len(dummy)))])

    for column_name in list(dummy):
        name_label = meta_land_use.LABEL3.loc[meta_land_use.CLC_CODE == column_name].to_string(index=False)
        name_label = name_label.strip().replace(' ', '_')
        dummy = dummy.rename(columns={column_name: name_label})
    
    return dummy, dummy_major          


def make_dummy_station_type(dataframe):
    dataframe[dataframe == 'industry'] = 'industrial'
    possibilities = ['background', 'traffic', 'industrial']
    exists = dataframe.unique()
    difference = pd.Series([item for item in possibilities if item not in exists])
    target = dataframe.append(pd.Series(difference))
    target = target.reset_index(drop=True)
    
    if len(difference) == 0:
       dummy = pd.get_dummies(target)
    else:       
       dummy = pd.get_dummies(target)
       dummy = dummy.drop(dummy.index[list(range(len(dummy) - len(difference), len(dummy)))])    
    return dummy[possibilities]          


def make_dummy_station_area(dataframe):
    possibilities = ['rural', 'urban', 'suburban', 'rural-regional', 'rural-remote', 'rural-nearcity', 'highmountain']
    exists = dataframe.unique()
    difference = pd.Series([item for item in possibilities if item not in exists])
    target = dataframe.append(pd.Series(difference))
    target = target.reset_index(drop=True)
    
    if len(difference) == 0:
       dummy = pd.get_dummies(target)
    else:       
       dummy = pd.get_dummies(target)
       dummy = dummy.drop(dummy.index[list(range(len(dummy) - len(difference), len(dummy)))])
    return dummy[possibilities]     
        

def make_dummy_smod(dataframe):
    possibilities = ['smod_rural', 'smod_urban', 'smod_suburban', 'smod_none']
    exists = dataframe.unique()
    difference = pd.Series([item for item in possibilities if item not in exists])
    target = dataframe.append(pd.Series(difference))
    target = target.reset_index(drop=True)
    
    if len(difference) == 0:
       dummy = pd.get_dummies(target)
    else:       
       dummy = pd.get_dummies(target)
       dummy = dummy.drop(dummy.index[list(range(len(dummy) - len(difference), len(dummy)))])
          
    return dummy[possibilities]      


def make_dummy_class(dataframe, number_clusters):
    possibilities = []
    for cluster_id in range(int(number_clusters)):
        possibilities.append(np.str(cluster_id) + '.0')
    exists = []
    for item in dataframe.unique():
        exists.append(np.str(item))

    difference = pd.Series([item for item in possibilities if item not in exists])
    target = dataframe.append(pd.Series(difference))
    target = target.reset_index(drop=True)
    
    if len(difference) == 0:
       dummy = pd.get_dummies(target, prefix='class_id')
    else:       
       dummy = pd.get_dummies(target, prefix='class_id')
       dummy = dummy.drop(dummy.index[list(range(len(dummy) - len(difference), len(dummy)))])
    
    list_ids = []
    for possible_id in possibilities:
        list_ids.append('class_id_' + possible_id)
        
    return dummy[list_ids]                  
  

def data_transform_preprocess_omi_filled(dataframe, model_identifier, qa=50):
    dataframe.loc[dataframe.Concentration <= 0, 'Concentration'] = np.nan
    dataframe.loc[dataframe.Concentration > dataframe.Concentration.quantile(q=0.99999), 'Concentration'] = np.nan
    dataframe.loc[dataframe.Concentration < dataframe.Concentration.quantile(q=0.00001), 'Concentration'] = np.nan
    traffic_mask = np.isnan(dataframe.rld)
    traffic_mask_2 = (dataframe.tfv < 0)
    dataframe.loc[traffic_mask_2, 'tfv'] = np.nan    
    dataframe.loc[traffic_mask, 'tfv'] = 0
    dataframe.loc[traffic_mask, 'rld'] = 0
    dataframe.loc[np.isnan(dataframe.lu), 'lu'] = 523

    processed_dataframe = pd.DataFrame()  
    processed_dataframe['Longitude'] = dataframe.lon
    processed_dataframe['Latitude'] = dataframe.lat
    
    if qa == 75:
        processed_dataframe['Certainty_dist'] = dataframe.hourly_certainty_dist_qa75
    else: 
        processed_dataframe['Certainty_dist'] = dataframe.hourly_certainty_dist
       
    processed_dataframe['10m_u-component_of_wind_speed'] = dataframe.u10
    processed_dataframe['10m_v-component_of_wind_speed'] = dataframe.v10
    processed_dataframe['10m_wind_speed'] = np.linalg.norm([dataframe.u10, dataframe.v10], axis=0)
    processed_dataframe['Temperature_2m'] = dataframe.t2m
    processed_dataframe['Solar_radiation'] = dataframe.cdir
    processed_dataframe['Total_precipitation'] = dataframe.tp    
    processed_dataframe['Boundary_layer_height'] = dataframe.blh    
    
    processed_dataframe['Digital_Elevation_Map'] = dataframe.dem
    processed_dataframe['Digital_Elevation_Map_wt_400m'] = dataframe.dem_hpf_400m
    processed_dataframe['Digital_Elevation_Map_wt_1km'] = dataframe.dem_hpf_1km
    processed_dataframe['Digital_Elevation_Map_wt_6km'] = dataframe.dem_hpf_6km
    processed_dataframe['Digital_Elevation_Map_wt_25km'] = dataframe.dem_hpf_25km
    processed_dataframe['Digital_Elevation_Map_wt_100km'] = dataframe.dem_hpf_100km

    processed_dataframe['Population_density_f'] = np.log1p(dataframe.popf_max)
    processed_dataframe['Population_density_d'] = dataframe.popf_dist
    processed_dataframe['Traffic_volume_f'] = np.log1p(dataframe.tfvf_max)
    processed_dataframe['Traffic_volume_d'] = dataframe.tfvf_dist    
    processed_dataframe['Road_length_density_f'] = np.log1p(dataframe.rldf_max)
    processed_dataframe['Road_length_density_d'] = dataframe.rldf_dist
    processed_dataframe['TNO_emission_f'] = np.log1p(dataframe.tnof_max)
    processed_dataframe['TNO_emission_d'] = dataframe.tnof_dist
    processed_dataframe['Land use'] = dataframe.lu
    processed_dataframe['Population_density'] = np.log1p(dataframe['pop'])
    processed_dataframe['Traffic_volume'] = np.log1p(dataframe.tfv)
    processed_dataframe['Road_length_density'] = np.log1p(dataframe.rld)    
    
    processed_dataframe['Month'] = dataframe.index.month
    processed_dataframe['Day'] = dataframe.index.day
    processed_dataframe['Hour'] = dataframe.index.hour
    processed_dataframe['weekday'] = (1 * (dataframe.index.dayofweek < 5) + 
                                     2 * (dataframe.index.dayofweek == 5) + 
                                     3 * (dataframe.index.dayofweek == 6))
    processed_dataframe['doy'] = dataframe.index.dayofyear
    
    processed_dataframe = processed_dataframe.replace([np.inf, -np.inf], np.nan)
    min_max_scaler = preprocessing.MinMaxScaler()    
    scale_data = min_max_scaler.fit(processed_dataframe)    
    df_train_minmax = min_max_scaler.fit_transform(processed_dataframe)
    processed_dataframe = pd.DataFrame(df_train_minmax, columns=list(processed_dataframe))
    processed_dataframe = processed_dataframe.set_index(dataframe.index)   
    pickle.dump(scale_data, open(os.path.join(root_directory, 'models', model_identifier, 'min_max_scaler_variables.sav'), 'wb'))    
  
    power_transformer = preprocessing.PowerTransformer()    
    if qa == 75:
        scale_no2 = power_transformer.fit(dataframe.hourly_no2_avg_filled_qa75.values.reshape(-1, 1))
        processed_dataframe['Sentinel5p_no2_fill'] = power_transformer.fit_transform(dataframe.hourly_no2_avg_filled_qa75.values.reshape(-1, 1))   
        pickle.dump(scale_no2, open(os.path.join(root_directory, 'models', model_identifier, 'pw_scaler_filled_sat.sav'), 'wb'))  
    else:
        scale_no2 = power_transformer.fit(dataframe.hourly_no2_avg_filled.values.reshape(-1, 1))
        processed_dataframe['Sentinel5p_no2_fill'] = power_transformer.fit_transform(dataframe.hourly_no2_avg_filled.values.reshape(-1, 1))   
        pickle.dump(scale_no2, open(os.path.join(root_directory, 'models', model_identifier, 'pw_scaler_filled_sat.sav'), 'wb'))  

    scale_no2 = power_transformer.fit(dataframe.Concentration.values.reshape(-1, 1))
    processed_dataframe['Airbase_Concentration'] = power_transformer.fit_transform(dataframe.Concentration.values.reshape(-1, 1))   
    pickle.dump(scale_no2, open(os.path.join(root_directory, 'models', model_identifier, 'pw_scaler_0.001percent.sav'), 'wb'))        
                           
    dummy, dummy_major = make_dummy_all(dataframe)
    dummy = dummy.set_index(dataframe.index)
    dummy_major = dummy_major.set_index(dataframe.index)
          
    training_dataframe = processed_dataframe    
    training_dataframe['AirQualityStationEoICode'] = dataframe['AirQualityStationEoICode'] 
    training_dataframe['AirQualityStationArea'] = dataframe['AirQualityStationArea'] 
    training_dataframe['AirQualityStationType'] = dataframe['AirQualityStationType'] 
    training_dataframe = pd.concat([training_dataframe, dummy], axis=1) 
    
    return training_dataframe.dropna()


def model_sort_features(training_dataframe, model_name='Model0'):
    switcher = {
        "Model1": training_dataframe.drop(columns=[
            'Population_density_f', 'Population_density_d', 'Traffic_volume_f', 'Traffic_volume_d', 
            'Road_length_density_f', 'Road_length_density_d', 'TNO_emission_f', 'TNO_emission_d', 
            'Month', 'Day', '10m_u-component_of_wind_speed', '10m_v-component_of_wind_speed', 'Land use'
        ]),
        "Model2": training_dataframe.drop(columns=[
            'Digital_Elevation_Map_wt_400m', 'Digital_Elevation_Map_wt_1km', 'Digital_Elevation_Map_wt_6km', 
            'Digital_Elevation_Map_wt_25km', 'Digital_Elevation_Map_wt_100km', 'Population_density', 
            'Traffic_volume', 'Road_length_density', 'Month', 'Day', '10m_u-component_of_wind_speed', 
            '10m_v-component_of_wind_speed', 'Land use'
        ]),
        "Model3": training_dataframe.drop(columns=[
            'Population_density_f', 'Population_density_d', 'Traffic_volume_f', 'Traffic_volume_d', 
            'Road_length_density_f', 'Road_length_density_d', 'TNO_emission_f', 'TNO_emission_d', 
            '10m_u-component_of_wind_speed', '10m_v-component_of_wind_speed', 'Land use',
            'Digital_Elevation_Map_wt_400m', 'Digital_Elevation_Map_wt_1km', 'Digital_Elevation_Map_wt_6km', 
            'Digital_Elevation_Map_wt_25km', 'Digital_Elevation_Map_wt_100km', 'Month', 'Day'
        ]),
        "Model4": training_dataframe.drop(columns=['Population_density', 'Traffic_volume', 'Road_length_density', 'doy']),
    }
    return switcher.get(model_name, "Provide new name")


def umap_clustering(training_dataframe):
    filename = 'umap_classification_stns_index_%d.pkl' % (cross_validation_id) 
    if not os.path.isfile(os.path.join(root_directory, 'models', model_identifier, filename)): 
        import umap.umap_ as umap
        from sklearn.decomposition import PCA
        import hdbscan
        
        temp = training_dataframe.groupby(training_dataframe.AirQualityStationEoICode).mean()
        features = temp.drop(columns={
            'Sentinel5p_no2_fill', 'Certainty_dist', 'Hour', 'weekday', 'doy', 
            'Airbase_Concentration', 'Longitude', 'Latitude'
        })
        
        cluster_count_std = 50
        while cluster_count_std > 30:
            low_dim_dataframe = PCA(n_components=5).fit_transform(features)
            embedding = umap.UMAP(n_neighbors=5, metric='correlation', min_dist=0.1, local_connectivity=3).fit_transform(low_dim_dataframe)
            hdbscan_labels = hdbscan.HDBSCAN(min_cluster_size=30).fit_predict(embedding)       
            dataframe_class = pd.DataFrame()
            dataframe_class['umap_1'] = embedding[:, 0]
            dataframe_class['umap_2'] = embedding[:, 1]
            dataframe_class['hdbscan'] = hdbscan_labels
            dataframe_class['hdbscan'] = 'class_' + dataframe_class['hdbscan'].apply(str)
            dataframe_class.hdbscan[dataframe_class['hdbscan'] == 'class_-1'] = str('not clustered')
            cluster_count_std = dataframe_class.groupby('hdbscan').count().std().umap_1
        
        features = features.reset_index()
        features = pd.concat([features, dataframe_class], axis=1) 
        features.to_pickle(os.path.join(root_directory, 'models', model_identifier, filename))   
    else:
        print('Stations are clustered and saved as cvID of %d: loading the saved file' % cross_validation_id)  
        features = pd.read_pickle(os.path.join(root_directory, 'models', model_identifier, filename))   
    return features


def split_train_test_umap_clustering_fixed_test(training_dataframe, train_size, root_path, model_id, identifier=0):
    filename = 'train_stns_fixed_test__train_size_%.1f_index_%d_%d.pkl' % (train_size, cross_validation_id, identifier)     
    
    if not os.path.isfile(os.path.join(root_path, 'models', model_id, filename)):  
        fname = 'umap_classification_stns_index_%d.pkl' % (cross_validation_id) 
        if not os.path.isfile(os.path.join(root_path, 'models', model_id, fname)): 
            features_old = umap_clustering(training_dataframe)
        else:
            features_old = pd.read_pickle(os.path.join(root_path, 'models', model_id, fname))
            
        fname = 'train_stns_umap_train_size_0.9_index_%d_%d.pkl' % (cross_validation_id, identifier) 
        train_stations_old = pd.read_pickle(os.path.join(root_path, 'models', model_id, fname)) 
        features = features_old[features_old['AirQualityStationEoICode'].isin(train_stations_old)].reset_index(drop=True)           
        test_stations = features_old['AirQualityStationEoICode'][~features_old['AirQualityStationEoICode'].isin(train_stations_old)].reset_index(drop=True)
        
        cluster_ids = features.hdbscan.unique()
        train_stations = []
        for cluster_id in cluster_ids:
            number_stations = len(features.loc[features['hdbscan'] == cluster_id])
            number_train_stations = int(number_stations * train_size)
            train_stations.append(features['AirQualityStationEoICode'].iloc[
                np.random.choice(features.loc[features['hdbscan'] == cluster_id].index, number_train_stations, replace=False)
            ])      
               
        train_stations = pd.concat(train_stations).sample(frac=1).reset_index(drop=True) 
        train_stations.to_pickle(os.path.join(root_path, 'models', model_id, filename))           
    else:
        fname = 'umap_classification_stns_index_%d.pkl' % (cross_validation_id) 
        features_old = pd.read_pickle(os.path.join(root_path, 'models', model_id, fname))       
        fname = 'train_stns_umap_train_size_0.9_index_%d_%d.pkl' % (cross_validation_id, identifier) 
        train_stations_old = pd.read_pickle(os.path.join(root_path, 'models', model_id, fname)) 
        test_stations = features_old['AirQualityStationEoICode'][~features_old['AirQualityStationEoICode'].isin(train_stations_old)].reset_index(drop=True)       
        train_stations = pd.read_pickle(os.path.join(root_path, 'models', model_id, filename))     
                       
    X_train, y_train = separate_target_features(training_dataframe[training_dataframe['AirQualityStationEoICode'].isin(train_stations)].sample(frac=1))
    X_test, y_test = separate_target_features(training_dataframe[~training_dataframe['AirQualityStationEoICode'].isin(train_stations)].sample(frac=1))
    X_valid, y_valid = separate_target_features(training_dataframe[training_dataframe['AirQualityStationEoICode'].isin(test_stations)].sample(frac=1))

    return X_train, X_test, y_train, y_test, X_valid, y_valid


def inverse_scale(prediction, model_id):
    scalers = pickle.load(open(os.path.join(root_directory, 'models', model_id, 'pw_scaler_0.001percent.sav'), 'rb'))       
    return scalers.inverse_transform(prediction)


def get_r2_on_real_values(exported_pipeline, features, target):
    y_pred_real = inverse_scale(exported_pipeline.predict(features).reshape(-1, 1), model_identifier)
    y_obs_real = inverse_scale(target.values.reshape(-1, 1), model_identifier)
    return metrics.r2_score(y_obs_real, y_pred_real)


def separate_target_features(training_dataframe):
    target = training_dataframe.Airbase_Concentration
    features = training_dataframe.drop(columns=[
        'Airbase_Concentration', 'AirQualityStationArea',
        'AirQualityStationEoICode', 'AirQualityStationType',
        'Longitude', 'Latitude'
    ])
    return features, target


def train_save_xgb_models_fixed_test_set(training_dataframe, train_size):
    features, target = separate_target_features(training_dataframe)
    X_train, X_test, y_train, y_test, X_valid, y_valid = split_train_test_umap_clustering_fixed_test(
        training_dataframe, train_size, root_directory, model_identifier
    )    

    exported_pipeline = XGBRegressor(
        tree_method='gpu_hist', 
        learning_rate=0.1, 
        max_depth=15, 
        min_child_weight=12, 
        n_estimators=100,
        n_jobs=12, 
        nthread=1, 
        subsample=0.9500000000000001
    )
    
    exported_pipeline.fit(X_train, y_train)    
    accuracy_test = get_r2_on_real_values(exported_pipeline, X_test, y_test)
    accuracy_train = get_r2_on_real_values(exported_pipeline, X_train, y_train)
    accuracy_all = get_r2_on_real_values(exported_pipeline, features, target)   
    accuracy_valid = get_r2_on_real_values(exported_pipeline, X_valid, y_valid)    

    filename = 'XGB_fixed_test_%s_train_size_%.1f_%.4f_%.4f_%.4f_%.4f_index_%d.sav' % (
        model_name, train_size, accuracy_test, accuracy_train, accuracy_all, accuracy_valid, cross_validation_id
    ) 
    print(filename)
    pickle.dump(exported_pipeline, open(os.path.join(root_directory, 'models', model_identifier, filename), 'wb'))
    

def main():
    description = textwrap.dedent("""\   
        This script is to train XGB models and to calculate cross-validation score for different different train/validation set that are selected based on umap clustering
        Here, there are several models using different features
        ---------------------------------------        
        Model1: Null model : Using data without extraction of particular features as other models
        Model2. Spatial features : Feature extraction of emission sources (max values and max distance in range [0,100])
        Model3. Spatial features + wavelet transform of DEM (digital elevation model) 
        Model4. dimensionality reduction  (rebate), remove some features 
        -------------------------------------
    """)
        
    parser = argparse.ArgumentParser(
        description=description, 
        epilog='',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('train_size', type=float, 
                       help='size of train set proportional to the all data (number between 0 and 1)')
    parser.add_argument('Model', type=str, 
                       help='model name (Model1, Model2, Model3, Model4)')
    parser.add_argument('qa', type=int, 
                       help='where to cut the training data; quality assurance level of 50 or75')
    args = parser.parse_args()

    global root_directory, model_name, model_identifier, cross_validation_id

    model_name = args.Model
    train_size = args.train_size
    qa_value = args.qa
    
    root_directory = '.'
    model_identifier = 'Full_features_' + str(qa_value)     
    cross_validation_id = 0

    if not os.path.isfile(os.path.join(root_directory, 'models', model_identifier, 'df_data.pkl')):
        dataframe = pd.read_pickle(os.path.join(root_directory, 'data', 'ROI1_full_features_qa75.pkl'))
        training_dataframe = data_transform_preprocess_omi_filled(dataframe, model_identifier, qa=qa_value)
        if not os.path.isdir(os.path.join(root_directory, 'models', model_identifier)):
            os.mkdir(os.path.join(root_directory, 'models', model_identifier)) 
        training_dataframe.to_csv(os.path.join(root_directory, 'models', model_identifier, 'df_data.csv'))  
        training_dataframe.to_pickle(os.path.join(root_directory, 'models', model_identifier, 'df_data.pkl'))   
    else:
        training_dataframe = pd.read_pickle(os.path.join(root_directory, 'models', model_identifier, 'df_data.pkl'))   

    training_dataframe_aggregated = model_sort_features(training_dataframe, model_name)    
    
    training_dataframe_aggregated = training_dataframe_aggregated.loc[
        ~((training_dataframe_aggregated.index.year == 2018) & 
          (training_dataframe_aggregated.index.month < 6))
    ]
    training_dataframe_aggregated = training_dataframe_aggregated.loc[
        ~((training_dataframe_aggregated.index.year == 2020) & 
          (training_dataframe_aggregated.index.month > 6))
    ]

    train_save_xgb_models_fixed_test_set(training_dataframe_aggregated, train_size)
      

if __name__ == '__main__':
    main()