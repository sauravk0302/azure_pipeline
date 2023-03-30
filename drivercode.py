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

if 'insurance dataset' not in ws.datasets:
    Dataset.File.upload_directory(src_dir='data',
                              target=DataPath(default_ds, 'insurance-data/')
                              )

    #Create a tabular dataset from the path on the datastore 
    tab_data_set = Dataset.Tabular.from_delimited_files(path=(default_ds, 'insurance-data/*.csv'))

    # Register the tabular dataset
    try:
        tab_data_set = tab_data_set.register(workspace=ws, 
                                name='insurance dataset',
                                description='insurance data',
                                tags = {'format':'CSV'},
                                create_new_version=True)
        print('Dataset registered.')
    except Exception as ex:
        print(ex)
else:
    print('Dataset already registered.')

# Create the environment
myenv_name = "Yash_AML_Env"

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
insurance_ds = ws.datasets.get("insurance dataset")

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

