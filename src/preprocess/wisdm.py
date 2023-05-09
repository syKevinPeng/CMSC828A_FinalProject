import sys, os, yaml
import pathlib
import pandas as pd
import numpy as np
from pathlib import Path
from utils.utils import get_logger

# https://archive.ics.uci.edu/ml/datasets/WISDM+Smartphone+and+Smartwatch+Activity+and+Biometrics+Dataset+
class WisdmProcessor():
    def __init__(self, config) -> None:
        self.config = config
        self.data_dir = pathlib.Path(self.config["in"]["dir"])
        if not self.data_dir.is_dir(): raise Exception(f"Input Path Not Found: {self.data_dir}")
        # load intersted label from config
        self.label = self.config["label"]
        self.label_dict:dict = self.config["label_converter"]
        self.uni_label = self.config["universal_label"]
        self.data_orgnization = "mean"
        self.logger = get_logger(self.config["out"]["dir"], "WisdmProc")
    
    def load_files(self):
        all_acc_data = []
        # read all accelermeter data
        for file_path in sorted((Path(self.data_dir)/"accel").glob('*.txt')):
            df = pd.read_csv(file_path, names = ["part_id", "label", "timestamp", "x", "y", "z"],lineterminator=";", low_memory=False)
            df['part_id'] = np.full(len(df), df.iloc[0,0])
            all_acc_data.append(df)
        all_acc_df = pd.concat(all_acc_data, axis = 0)
        
        all_acc_df.loc[:, "timestamp"] = pd.to_datetime(all_acc_df.loc[:, "timestamp"]).round("10ms")
        # all_gyro_df.loc[:, "timestamp"] = pd.to_datetime(all_gyro_df.loc[:, "timestamp"]).round("10ms")
        
        all_acc_df = all_acc_df.dropna(axis=0, how="any")
        # all_gyro_df = all_gyro_df.dropna(axis=0, how="any")
        # only keep 4000 entries of label D

        all_acc_df = all_acc_df.sort_values(by=["part_id", "timestamp"])
        all_acc_df = all_acc_df.reset_index(drop=True)
        return all_acc_df
    
    def label_converter(self, label):
        return self.label_dict[label]

    def preprocess_df(self, all_df):
        # normalization
        name_col = ["x", "y", "z"]
        # for col in name_col:
        #     all_df.loc[:, col] = (all_df[col]-all_df[col].mean())/all_df[col].std()
        activity_list = []
        for label in self.label:
            act_df = all_df[all_df["label"] == label].loc[:, ["part_id", "timestamp"] + name_col]
            placeholder= np.zeros((len(act_df),len(self.uni_label)))
            current_label = self.label_converter(label)
            label_idx = self.uni_label.index(current_label)
            placeholder[:,label_idx] = np.ones((len(act_df)))
            act_block = np.concatenate([act_df.to_numpy(), placeholder], axis = 1)
            col_name = ["part_id", 'timestamp'] + name_col + self.uni_label
            act_df = pd.DataFrame(act_block, columns =col_name)
            act_df = act_df.reset_index(drop=True)
            activity_list.append(act_df)
        all_df = pd.concat(activity_list, axis = 0,)
        # perform resampling for each of the participants with part_id
        # resampled_df = []
        # for part_id in all_df["part_id"].unique():
        #     self.logger.info(f"Resampling data of part_id {part_id}")
        #     input_df = all_df[all_df["part_id"] == part_id]
        #     input_df[name_col] = input_df[name_col].apply(pd.to_numeric)
        #     resampled = self.resampling(input_df)
        #     resampled_df.append(resampled)
        # resampled_df = all_df
        # grouped_df = pd.concat(resampled_df, axis = 0)
        # grouped_df = grouped_df.reset_index(drop=True)
        grouped_df = all_df

        # D_acc_df = all_acc_df[all_acc_df["label"] == "D"]
        # all_acc_df = all_acc_df[all_acc_df["label"] != "D"]
        # all_acc_df = pd.concat([all_acc_df, D_acc_df.iloc[:4000, :]], axis = 0)
        return grouped_df
    
    def resampling(self, df):
        df = df.set_index("timestamp")
        df = df.drop(columns=["part_id"])
        # limit size of data if needed
        # df = df.iloc[:2000,:]
        
        uni_time_freq = pd.Timedelta(value=1/24, unit='min')
        user_df= df.resample(uni_time_freq).mean()
        user_df = user_df.interpolate(method='time')

        if self.data_orgnization == "group":
            agg_df = user_df.resample("20s").agg(list).reset_index()[1:]
            for label_col in self.uni_label:
                    agg_df[label_col] = agg_df[label_col].apply(
                        lambda x: sum(x)
                    )
            label = agg_df.iloc[:,-(len(self.uni_label)):]
            m = np.zeros_like(label.values)
            m[np.arange(len(label)), label.values.argmax(1)] = 1
            agg_df.iloc[:,-6:] = pd.DataFrame(m, columns = label.columns).astype(int)
        
        elif self.data_orgnization == 'none':
            agg_df = user_df.reset_index()
            agg_df = agg_df.dropna(axis=0, how="any")
        
        elif self.data_orgnization == 'mean':
            agg_df = user_df.resample("1s").mean().reset_index()
            labels = agg_df.iloc[:,-(len(self.uni_label)):].to_numpy()
            idx = labels.argmax(axis=1)
            labels = (idx[:,None] == np.arange(labels.shape[1])).astype(float)
            agg_df.iloc[:,-(len(self.uni_label)):] = pd.DataFrame(labels, columns = self.uni_label).astype(int)
        
        else:
            raise ValueError("data_orgnization should be one of ['group', 'none', 'mean']")
        return agg_df

    
    def save_df(self, df: pd.DataFrame, save_dir):
        df.to_csv(save_dir, index = False)
        print(f"Saved preprocessed wisdm to {save_dir}")


    def preprocess(self):
        merged_df = self.load_files()
        preprocessed_df = self.preprocess_df(merged_df)
        return preprocessed_df