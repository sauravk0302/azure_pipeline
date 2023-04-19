import datetime as dt
import pandas as pd
from azureml.core import Workspace
from azureml.core import Dataset
from azureml.data.datapath import DataPath
ws = Workspace.from_config()

default_ds = ws.get_default_datastore()
tab_data_set = Dataset.Tabular.from_delimited_files(path=(default_ds, 'insurance-data/*.csv'))

    # Register the tabular dataset
    # try:
tab_data_set = tab_data_set.register(workspace=ws, 
                                name='insurance dataset base',
                                description='insurance data base',
                                tags = {'format':'CSV'},
                                create_new_version=True)
print('Dataset registered.')
print('Generating simulated data...')
cluster_name = "saurav-compute-cluster"
# Load the smaller of the two data files
data = pd.read_csv('data/insurance_claims.csv')

# We'll generate data for the past 6 weeks
weeknos = reversed(range(6))

file_paths = []
for weekno in weeknos:
    
    # Get the date X weeks ago
    data_date = dt.date.today() - dt.timedelta(weeks=weekno)
    
    # Modify data to ceate some drift
    data['age'] = round(data['age'] * 1.3).astype(int)
    data['months_as_customer'] = data['months_as_customer'] * 1.1
    
    # Save the file with the date encoded in the filename
    file_path = 'data/target_data/insurance_claims_{}.csv'.format(data_date.strftime("%Y-%m-%d"))
    data.to_csv(file_path)
    file_paths.append(file_path)

# Upload the files
path_on_datastore = 'insurance-target'
default_ds.upload_files(files=file_paths,
                       target_path=path_on_datastore,
                       overwrite=True,
                       show_progress=True)

# Use the folder partition format to define a dataset with a 'date' timestamp column
partition_format = path_on_datastore + '/insurance_claims_{date:yyyy-MM-dd}.csv'
target_data_set = Dataset.Tabular.from_delimited_files(path=(default_ds, path_on_datastore + '/*.csv'),
                                                       partition_format=partition_format)

# Register the target dataset
print('Registering target dataset...')
target_data_set = target_data_set.with_timestamp_columns('date').register(workspace=ws,
                                                                          name='insurance target',
                                                                          description='insurance target data',
                                                                          tags = {'format':'CSV'},
                                                                          create_new_version=True)

print('Target dataset registered!')

#DriftDetector
from azureml.datadrift import DataDriftDetector

# set up feature list
features = ['age', 'months_as_customer']

# set up data drift detector
monitor = DataDriftDetector.create_from_datasets(ws, 'mslearn-insurance-drift_8', tab_data_set, target_data_set,
                                                      compute_target=cluster_name, 
                                                      frequency='Week', 
                                                      feature_list=features, 
                                                      drift_threshold=.3, 
                                                      latency=24)
monitor

# from azureml.widgets import RunDetails

backfill = monitor.backfill(dt.datetime.now() - dt.timedelta(weeks=6), dt.datetime.now())

# RunDetails(backfill).show()
backfill.wait_for_completion()
     
drift_metrics = backfill.get_metrics()
for metric in drift_metrics:
    print(metric, drift_metrics[metric])     
