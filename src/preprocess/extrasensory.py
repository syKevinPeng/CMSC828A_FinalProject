import glob
from logging import exception 
from tqdm import tqdm 
import pandas as pd
from datetime import datetime
import sys 
import numpy as np
from pathlib import Path
import logging

class ExtrasensoryProcessor(): 
    def __init__(self, config): 
        self.config = config 
        
    def preprocess(self): 
        csv_files = glob.glob(f"{self.config['in']['dir']}/*.csv", recursive = True)

        if len(csv_files) == 0: raise Exception(f"No csv file found:{self.config['in']['dir']}")
        all_df = []
        for file in csv_files:
            df = pd.read_csv(file)
            x_all,y_all = self.load_and_clean(df)
            # select and convert
            y_all = y_all[self.config["labels"]]
            y_all = self.convert_label(y_all)
            # normalization
            for col in x_all.columns[1:]:
                x_all[col] = (x_all[col] - x_all[col].mean()) / x_all[col].std()
            x_all.columns = ['timestamp', 'x', 'y', 'z', 'ro_xy', 'ro_xz', 'ro_yz']
            df = pd.concat([x_all,y_all], axis=1)
            df.dropna(axis = 0, how = 'any',inplace = True)

            # resampling
            df = self.resampling(df)
            all_df.append(df)
        all_df = pd.concat(all_df)
        return all_df
    
    def save_df(self, df: pd.DataFrame, save_dir):
        df.to_csv(save_dir, index=False)
        print(f"processed dataframe is saved at {save_dir}")
            
    # Since the extrasensory data is grouped into 1 min interval, we does not perform any resampling here.
    def resampling(self, df):
        # df = df.set_index('timestamp')
        df.loc[:, "timestamp"] = pd.to_datetime(
            df.loc[:, "timestamp"],  unit='s')
        # df = df.resample('1min').mean()
        # df = df.reset_index()
        return df

    # This this a code loading and cleaning function. As this old code is from phase 1,I did some changes to it. However,
    # there is clealy some improvement can be done here.
    def load_and_clean(self, users_df):
        label_names = self.config["labels"]
        # get needed sensory name
        users_df = users_df.fillna(0)
        # split X and Y
        x_columns = ['timestamp', 'watch_acceleration:3d:mean_x', 'watch_acceleration:3d:mean_y', 'watch_acceleration:3d:mean_z', 
            'watch_acceleration:3d:ro_xy', 'watch_acceleration:3d:ro_xz','watch_acceleration:3d:ro_yz']
        X = users_df[x_columns]
        Y = users_df[label_names]
        # Prepare Y target lables for training
        label_copy = label_names[:]
        
        has_uuid = False
        if 'label:uuid' in Y: 
            label_copy.remove("label:uuid")
            uuid = Y['label:uuid']
            has_uuid = True 
        
        len_before = len(Y)
        
        # Read the binary label values, and the 'missing label' indicators:
        trinary_labels_mat = users_df[label_copy]; # This should have values of either 0., 1. or NaN
        M = np.isnan(trinary_labels_mat); # M is the missing label matrix
        Y = np.where(M,0,trinary_labels_mat) > 0.; # Y is the label matrix
        
        y_df = pd.DataFrame(Y)
        assert len_before == len(y_df)
        
        if has_uuid: 
            y_df['label:uuid'] = uuid
        y_df.rename(columns=dict(enumerate(label_names, 0)), inplace = True)
        return X,y_df

    def prepare_extrasensory(self, dir, sensor = ['acc'], exceptions=None): 
        # ## Get all Extrasensory user file data into one dataframe
        
        i = 0 
        csv_files = glob.glob(f"{dir}/*.csv")
        if len(csv_files) == 0: raise Exception(f'extrasensory data does not found: {dir}')
        list_user_files = [] 
        for file in tqdm(csv_files):
            uuid = file.split("/")[-1].split(".")[0]
            if exceptions and uuid in exceptions: 
                print(f"leaving out: {uuid}")
                continue 
            
            df = pd.read_csv(file)
            df.insert(loc=len(df.columns) - 1, column='label:uuid', value=uuid)
        
            list_user_files.append(df)
            
            i += 1 
        
        print(f"read in {len(list_user_files)} user files")
        all_users_data = pd.concat(list_user_files, axis=0, ignore_index=True)

        (X_all,Y_all,feature_names,label_names) = self. prepare_X_Y_for_ML(all_users_data)
        
        # ## Get all Watch acceleration features
        # features_of_selected_sensors = project_features_to_selected_sensors(feature_names, ['WAcc'])
        # features_of_selected_sensors = [f for f in features_of_selected_sensors if "3d" in f]
        if len(sensor) == 1 and sensor[0] == 'acc':
            features_of_selected_sensors = ['watch_acceleration:3d:ro_xy', 'watch_acceleration:3d:ro_xz', 'watch_acceleration:3d:ro_yz']
        elif (len(sensor) == 2) and ('acc' in sensor) and ("gyro" in sensor):
            features_of_selected_sensors = ['watch_acceleration:3d:ro_xy', 'watch_acceleration:3d:ro_xz', 'watch_acceleration:3d:ro_yz','proc_gyro:3d:ro_xy', 'proc_gyro:3d:ro_xz', 'proc_gyro:3d:ro_yz']
        else:
            raise Exception(f"Unable to recognize sensor list:{sensor}")
        X_all = X_all[features_of_selected_sensors]
        
        # normalization
        print(type(X_all))
        for col in X_all.columns:
            X_all[col] = (X_all[col] - X_all[col].mean()) / X_all[col].std()
        return X_all, Y_all
    
    def get_features_from_data(self, users_df):
        for (ci,col) in enumerate(users_df.columns):
            if col.startswith('label:'):
                first_label_ind = ci
                break
        pass
        feature_names = users_df.columns[1:first_label_ind]
        # if 'label:uuid' in users_df.columns: 
        #     feature_names.append(users_df.columns[users_df.columns.get_loc('label:uuid')])
        return np.array(feature_names)

    def project_features_to_selected_sensors(self, feature_names,sensors_to_use):

        feature_names_arr = []
        for sensor in sensors_to_use:
            if sensor == 'Acc':
                for feature in feature_names:
                    #print (type(feature))
                    if (feature.startswith('raw_acc')):
                        feature_names_arr.append(feature)
            elif sensor == 'WAcc':
                for feature in feature_names:
                    if (feature.startswith('watch_acceleration')):
                        feature_names_arr.append(feature)
            elif sensor == 'Gyro':
                for feature in feature_names:
                    if (feature.startswith('proc_gyro')):
                        feature_names_arr.append(feature)
            elif sensor == 'Magnet':
                for feature in feature_names:
                    if (feature.startswith('raw_magnet')):
                        feature_names_arr.append(feature)
            elif sensor == 'Compass':
                for feature in feature_names:
                    if (feature.startswith('watch_heading')):
                        feature_names_arr.append(feature)
            elif sensor == 'Loc':
                for feature in feature_names:
                    if (feature.startswith('location')):
                        feature_names_arr.append(feature)
            elif sensor == 'Aud':
                for feature in feature_names:
                    if (feature.startswith('audio_naive')):
                        feature_names_arr.append(feature)
            elif sensor == 'AP':
                for feature in feature_names:
                    if (feature.startswith('audio_properties')):
                        feature_names_arr.append(feature)
            elif sensor == 'PS':
                for feature in feature_names:
                    if (feature.startswith('discrete')):
                        feature_names_arr.append(feature)
            elif sensor == 'LF':
                for feature in feature_names:
                    if (feature.startswith('lf_measurements')):
                        feature_names_arr.append(feature)
                        
        return feature_names_arr
    
    # convert label according to the config mapping
    def convert_label(self,  y_all):
        map_config = self.config['label_converter']
        old_column_name = y_all.columns
        # remove "label:" and prefix such as "FIX" and "OR" in the label name
        new_column_name = old_column_name.str.replace("label:", "es__")
        new_column_name = new_column_name.str.replace("FIX_", "")
        new_column_name = new_column_name.str.replace("OR_", "").tolist()
        new_column_name = [i.lower() for i in new_column_name]
        y_all.columns = new_column_name
        for activity in map_config:  
            col_name = f"{activity}"
            expr = map_config[activity] 
            full_expr = f"{col_name} = {expr}"
            y_all = y_all.eval(full_expr)
        y_all_df = y_all[list(map_config.keys())]
        
        if 'es__uuid' in y_all: 
            y_all_df['label:uuid'] = y_all['es__uuid']
            
        return y_all_df
    
    # choosing a fixed number of samples that are closest to the centroid of each old class
    # input: preprocessed data frame
def herd_selection(df:pd.DataFrame, output_dir:Path, logger=None):
    # select a fix number of samples that are closest to the centroid of each old class
    # NUMBER OF SAMPLES TO SELECT
    N = 200 
    if logger is None:
        logger = logging.getLogger(__name__)
    logger.info('------ Herd Selection ------')
    logger.info(f'number of reserverd samples for each class to reserve: {N}')
    classes = df.columns[7:].to_numpy()
    logger.info(f'total number of class: {len(classes)}')
    reserved_samples = {}
    for label in classes:
        # select all rows with the columns=label is true
        label_df = df[df[label] == True]
        print(f'length of {label} is {len(label_df)}')
        # get the centroid of the label
        label_df_feature = label_df[['x','y','z','ro_xy','ro_xz','ro_yz']]
        centroid = label_df_feature.mean(axis=0)
        # calculate the distance between each row and the centroid
        label_df['distance'] = label_df_feature.apply(lambda row: np.linalg.norm(row - centroid), axis=1)
        # sort the rows by distance
        label_df = label_df.sort_values(by=['distance'])
        # select the first N rows
        label_df = label_df[label_df[label]==True].iloc[:N] # store positive samples
        label_df = label_df.drop(columns=['distance'])
        # reset index
        label_df = label_df.reset_index(drop=True)
        # update the df
        reserved_samples[label] = label_df
    # concat all the dataframes
    reserved_df = pd.concat(reserved_samples.values(), ignore_index=True)
    # save the df to csv
    reserved_df.to_csv(output_dir/'herd_samples.csv', index=False)
    logger.info(f'herd select data is saved to {output_dir}/herd_samples.csv')
    logger.info('------ Herd Selection Done ------')


if __name__ == "__main__":
    processed_csv = '/fs/class-projects/spring2023/cmsc828a/c828ag04/CMSC828A_FinalProject/datasets/preprocessed/extrasensory/preprocessed_es.csv'
    output_dir = Path('/fs/class-projects/spring2023/cmsc828a/c828ag04/CMSC828A_FinalProject/datasets/preprocessed/extrasensory')
    df = pd.read_csv(processed_csv)
    herd_selection(df, output_dir)
