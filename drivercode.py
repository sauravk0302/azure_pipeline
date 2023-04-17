import os
import azureml.core
from azureml.core import Workspace
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core import Environment
from azureml.core.runconfig import RunConfiguration
from azureml.data import OutputFileDatasetConfig
from azureml.pipeline.core import Pipeline
from azureml.pipeline.steps import PythonScriptStep
from azureml.core import Experiment
from azureml.core import Model

# Load the workspace from the saved config file
ws = Workspace.from_config()
print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))

from azureml.core import Dataset
from azureml.data.datapath import DataPath

default_ds = ws.get_default_datastore()

# if 'insurance dataset' not in ws.datasets:
Dataset.File.upload_directory(src_dir='data',
                              target=DataPath(default_ds, 'insurance-data/')
                              )

    #Create a tabular dataset from the path on the datastore 
tab_data_set = Dataset.Tabular.from_delimited_files(path=(default_ds, 'insurance-data/*.csv'))

    # Register the tabular dataset
    # try:
tab_data_set = tab_data_set.register(workspace=ws, 
                                name='insurance dataset base',
                                description='insurance data base',
                                tags = {'format':'CSV'},
                                create_new_version=True)
print('Dataset registered.')
#     except Exception as ex:
#         print(ex)
# else:
#     print('Dataset already registered.')
 
# Create the environment
myenv_name = "Yash_AML_Env_1"

# Create a folder for the pipeline step files
experiment_folder = 'insurance_pipeline'
os.makedirs(experiment_folder, exist_ok=True)

print(experiment_folder)

cluster_name = "saurav-compute-cluster"

try:
    # Check for existing compute target
    pipeline_cluster = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    # If it doesn't already exist, create it
    try:
        compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS11_V2', max_nodes=2)
        pipeline_cluster = ComputeTarget.create(ws, cluster_name, compute_config)
        pipeline_cluster.wait_for_completion(show_output=True)
    except Exception as ex:
        print(ex)    

# Create a Python environment for the experiment (from a .yml file)
experiment_env = Environment.from_conda_specification(myenv_name, experiment_folder + "/experiment_env.yml")

# Register the environment 
experiment_env.register(workspace=ws)
registered_env = Environment.get(ws, myenv_name)

# Create a new runconfig object for the pipeline
pipeline_run_config = RunConfiguration()

# Use the compute you created above. 
pipeline_run_config.target = pipeline_cluster

# Assign the environment to the run configuration
pipeline_run_config.environment = registered_env

print ("Run configuration created.")

# Get the training dataset
insurance_ds = ws.datasets.get("insurance dataset base")

# Create an OutputFileDatasetConfig (temporary Data Reference) for data passed from step 1 to step 2
prepped_data = OutputFileDatasetConfig("prepped_data")

# Step 1, Run the data prep script
prep_step = PythonScriptStep(name = "Prepare Data",
                                source_directory = experiment_folder,
                                script_name = "prep_insurance.py",
                                arguments = ['--input-data', insurance_ds.as_named_input('raw_data'),
                                             '--prepped-data', prepped_data],
                                compute_target = pipeline_cluster,
                                runconfig = pipeline_run_config,
                                allow_reuse = True)

# Step 2, run the training script
train_step = PythonScriptStep(name = "Train and Register Model",
                                source_directory = experiment_folder,
                                script_name = "train_insurance.py",
                                arguments = ['--training-data', prepped_data.as_input()],
                                compute_target = pipeline_cluster,
                                runconfig = pipeline_run_config,
                                allow_reuse = True)

print("Pipeline steps defined")

# Construct the pipeline
pipeline_steps = [prep_step, train_step]
pipeline_new = Pipeline(workspace=ws, steps=pipeline_steps)
print("Pipeline is built.")

# Create an experiment and run the pipeline
experiment_new = Experiment(workspace=ws, name = 'AML-Insurance-pipeline')
pipeline_run = experiment_new.submit(pipeline_new, regenerate_outputs=True)
print("Pipeline submitted for execution.")
# RunDetails(pipeline_run).show()
pipeline_run.wait_for_completion(show_output=True)


for run in pipeline_run.get_children():
    print(run.name, ':')
    metrics = run.get_metrics()
    for metric_name in metrics:
        print('\t',metric_name, ":", metrics[metric_name])

for model in Model.list(ws):
    print(model.name, 'version:', model.version)
    for tag_name in model.tags:
        tag = model.tags[tag_name]
        print ('\t',tag_name, ':', tag)
    for prop_name in model.properties:
        prop = model.properties[prop_name]
        print ('\t',prop_name, ':', prop)
    print('\n')        

# Publish the pipeline from the run
published_pipeline = pipeline_run.publish_pipeline(
    name="insurance-training-pipeline", description="Trains insurance model", version="1.0")

published_pipeline

rest_endpoint = published_pipeline.endpoint
print(rest_endpoint)
     

from azureml.core.authentication import InteractiveLoginAuthentication

interactive_auth = InteractiveLoginAuthentication()
auth_header = interactive_auth.get_authentication_header()
print("Authentication header ready.")
     
import requests

experiment_name = 'AML-Insurance-pipeline'

rest_endpoint = published_pipeline.endpoint
response = requests.post(rest_endpoint, 
                         headers=auth_header, 
                         json={"ExperimentName": experiment_name})
run_id = response.json()["Id"]
run_id     

from azureml.pipeline.core.run import PipelineRun

published_pipeline_run = PipelineRun(ws.experiments[experiment_name], run_id)
published_pipeline_run.wait_for_completion(show_output=True)
###############################################################
# import datetime as dt
# import pandas as pd

# print('Generating simulated data...')

# # Load the smaller of the two data files
# data = pd.read_csv('data/insurance_claims.csv')

# # We'll generate data for the past 6 weeks
# weeknos = reversed(range(6))

# file_paths = []
# for weekno in weeknos:
    
#     # Get the date X weeks ago
#     data_date = dt.date.today() - dt.timedelta(weeks=weekno)
    
#     # Modify data to ceate some drift
#     data['age'] = round(data['age'] * 1.3).astype(int)
#     data['months_as_customer'] = data['months_as_customer'] * 1.1
    
#     # Save the file with the date encoded in the filename
#     file_path = 'data/target_data/insurance_claims_{}.csv'.format(data_date.strftime("%Y-%m-%d"))
#     data.to_csv(file_path)
#     file_paths.append(file_path)

# # Upload the files
# path_on_datastore = 'insurance-target'
# default_ds.upload_files(files=file_paths,
#                        target_path=path_on_datastore,
#                        overwrite=True,
#                        show_progress=True)

# # Use the folder partition format to define a dataset with a 'date' timestamp column
# partition_format = path_on_datastore + '/insurance_claims_{date:yyyy-MM-dd}.csv'
# target_data_set = Dataset.Tabular.from_delimited_files(path=(default_ds, path_on_datastore + '/*.csv'),
#                                                        partition_format=partition_format)

# # Register the target dataset
# print('Registering target dataset...')
# target_data_set = target_data_set.with_timestamp_columns('date').register(workspace=ws,
#                                                                           name='insurance target',
#                                                                           description='insurance target data',
#                                                                           tags = {'format':'CSV'},
#                                                                           create_new_version=True)

# print('Target dataset registered!')

# #DriftDetector
# from azureml.datadrift import DataDriftDetector

# # set up feature list
# features = ['age', 'months_as_customer']

# # set up data drift detector
# monitor = DataDriftDetector.create_from_datasets(ws, 'mslearn-insurance-drift_2', tab_data_set, target_data_set,
#                                                       compute_target=cluster_name, 
#                                                       frequency='Week', 
#                                                       feature_list=features, 
#                                                       drift_threshold=.3, 
#                                                       latency=24)
# monitor

# # from azureml.widgets import RunDetails

# backfill = monitor.backfill(dt.datetime.now() - dt.timedelta(weeks=6), dt.datetime.now())

# # RunDetails(backfill).show()
# backfill.wait_for_completion()
     
# drift_metrics = backfill.get_metrics()
# for metric in drift_metrics:
#     print(metric, drift_metrics[metric])     
