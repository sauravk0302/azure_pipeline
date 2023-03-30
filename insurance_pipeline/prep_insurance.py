# Import libraries
import os
import argparse
import pandas as pd
import numpy as np
import scipy.stats as stats 
from azureml.core import Run
from sklearn.preprocessing import LabelEncoder

# Get parameters
parser = argparse.ArgumentParser()
parser.add_argument("--input-data", type=str, dest='raw_dataset_id', help='raw dataset')
parser.add_argument('--prepped-data', type=str, dest='prepped_data', default='prepped_data', help='Folder for results')
args = parser.parse_args()
prep_folder = args.prepped_data

# Get the experiment run context
run = Run.get_context()

# load the data (passed as an input dataset)
print("Loading Data...")
df = run.input_datasets['raw_data'].to_pandas_dataframe()
row_count = (len(df))
run.log('raw_rows', row_count)

def Preprocessing(df):
    """Data Pre-processing"""
    df = df.replace('?',np.NaN)
    df['collision_type'].fillna(df['collision_type'].mode()[0], inplace = True)
    df['property_damage'].fillna('NO', inplace = True)
    df['police_report_available'].fillna('NO', inplace = True)
    df.drop(['_c39'], axis=1, inplace=True)    
    numeric_data = df._get_numeric_data()
    cat_data = df.select_dtypes(include=['object'])

    for c in cat_data:
        lbl = LabelEncoder()
        lbl.fit(cat_data[c].values)
        cat_data[c] = lbl.transform(cat_data[c].values)
    clean_data = pd.concat([numeric_data,cat_data],axis=1)
    return clean_data

dataPrep = Preprocessing(df)

#Log processed rows
row_count = (len(df))
run.log('processed_rows', row_count)

# Save the prepped data
print("Saving Data...")
os.makedirs(prep_folder, exist_ok=True)
save_path_1 = os.path.join(prep_folder,'prep_data.csv')
dataPrep.to_csv(save_path_1, index=False, header=True)

# End the run
run.complete()
