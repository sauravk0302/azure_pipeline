# Import libraries
from azureml.core import Run, Model
import pandas as pd
import numpy as np
import joblib
import os
import argparse, joblib
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# Get parameters
parser = argparse.ArgumentParser()
parser.add_argument("--training-data", type=str, dest='training_data', help='training data')
args = parser.parse_args()
training_data = args.training_data

# Get the experiment run context
run = Run.get_context()

# load the prepared data file in the training folder
print("Loading Data...")
file_path = os.path.join(training_data,'prep_data_new.csv')
data_prep = pd.read_csv(file_path)


# Get parameters
parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int)
parser.add_argument("--min_samples_leaf", type=int)
parser.add_argument("--datafolder", type=str)
args, unknown = parser.parse_known_args()
ne = args.n_estimators
msl = args.min_samples_leaf

print(ne, msl)

X = data_prep.drop("fraud_reported",axis=1)
print(X)
y=data_prep["fraud_reported"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1234)
#smote

sm = SMOTE(random_state = 2)
X_train, y_train= sm.fit_resample(X_train, y_train.ravel())

# Baseline Random forest based Model
# rfc = RandomForestClassifier(n_estimators=ne, min_samples_leaf=msl)
rfc = RandomForestClassifier(n_estimators=16,min_samples_leaf=5)

rfcg = rfc.fit(X_train, y_train) # fit on training data
Y_predict = rfcg.predict(X_test)

# Get the probability score - Scored Probabilities
Y_prob = rfcg.predict_proba(X_test)[:, 1]

# Get Confusion matrix and the accuracy/score - Evaluate

cm    = confusion_matrix(y_test, Y_predict)
accuracy = accuracy_score(y_test, Y_predict)

# Create the confusion matrix dictionary
cm_dict = {"schema_type": "confusion_matrix",
           "schema_version": "v1",
           "data": {"class_labels": ["N", "Y"],
                    "matrix": cm.tolist()}
           }

run.log("TotalObservations", len(data_prep))
run.log_confusion_matrix("ConfusionMatrix", cm_dict)
run.log("Accuracy", accuracy)

# Save the trained model in the outputs folder
print("Saving model...")
os.makedirs('outputs', exist_ok=True)
model_file = os.path.join('outputs', 'insurance_model.pkl')
joblib.dump(value=rfc, filename=model_file)

# Register the model
print('Registering model...')
Model.register(workspace=run.experiment.workspace,
               model_path = model_file,
               model_name = 'insurance_model_pipeline',
               tags={'Training context':'Pipeline'},
               properties={'Accuracy': np.float(accuracy)})


run.complete()
