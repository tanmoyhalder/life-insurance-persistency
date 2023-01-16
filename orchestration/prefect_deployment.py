import os
import warnings
warnings.simplefilter("ignore", UserWarning)

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

from datetime import datetime as dt
from datetime import timedelta
import pickle

import mlflow

from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
from prefect.filesystems import LocalFileSystem
from prefect.deployments import Deployment
from prefect.orion.schemas.schedules import IntervalSchedule

from feature_engine import encoding as ce
from feature_engine import imputation as mdi
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

import xgboost as xgb

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score



INPUT_FILEPATH = 'data'
INPUT_FILENAME = 'input_data.parquet'
INDEX_COL = 'policy_number'
COLS_TO_REM = ['proposal_received_date','policy_issue_date', 'zipcode', 'county', 'state', 'agent_code', 'agent_dob', 'agent_doj']
MISSING_COLS = ['agent_persistency']
ONE_HOT_COLS = ['owner_gender', 'marital_status', 'smoker', 'medical', 'education', 'occupation', 'payment_freq',  
                'agent_status', 'agent_education']

FEATURES = ['owner_age', 'owner_gender', 'marital_status', 'num_nominee', 'smoker',
       'medical', 'education', 'occupation', 'experience', 'income',
       'negative_zipcode', 'family_member', 'existing_num_policy',
       'has_critical_health_history', 'policy_term', 'payment_freq',
       'annual_premium', 'sum_insured', 'agent_status', 'agent_education',
       'agent_age', 'agent_tenure_days', 'agent_persistency',
       'last_6_month_submissions', 'average_premium', 'is_reinstated',
       'prev_persistency', 'num_complaints', 'target_completion_perc',
       'has_contacted_in_last_6_months', 'credit_score',
       'time_to_issue', 'prem_to_income_ratio']

TARGET = 'lapse'
RANDOM_STATE = 786
TEST_SIZE = 0.3


@task
def read_data(INPUT_FILEPATH, INPUT_FILENAME) -> pd.DataFrame:
    input_df = pd.read_parquet(os.path.join(INPUT_FILEPATH, INPUT_FILENAME),engine = 'pyarrow')
    input_df = input_df.set_index(INDEX_COL)
    print(input_df.shape)
    print(input_df['lapse'].value_counts()/len(input_df)*100)
    
    return input_df

@task
def create_features(df) -> pd.DataFrame:
    df['time_to_issue'] = (df['policy_issue_date'] - df['proposal_received_date']).dt.days
    df['prem_to_income_ratio'] = np.where(df['income'] == 0, 0, (df['annual_premium']/df['income']))
    print(df.shape)

    return df

@task
def clean_data(df)  -> pd.DataFrame:
    df = df.drop(COLS_TO_REM, axis = 1)
    print(df.shape)

    return df

@task
def crate_train_test(df) -> pd.DataFrame:

    X_train, X_test, y_train, y_test = train_test_split(df[FEATURES],
                                                    df[TARGET],
                                                    test_size= TEST_SIZE,
                                                    random_state = RANDOM_STATE, 
                                                    shuffle = True,
                                                    stratify = df[TARGET])
    
    print(X_train.shape, X_test.shape)

    model_input_pipe = Pipeline([
    ('imputer_num', mdi.MeanMedianImputer(imputation_method = 'median', variables = MISSING_COLS)), 
    ('onehot_encoder', ce.OneHotEncoder(top_categories=None,
                                        variables= ONE_HOT_COLS,
                                        drop_last=True)),
    ('normalisation', StandardScaler())])

    X_train_trf = model_input_pipe.fit_transform(X_train)
    X_test_trf = model_input_pipe.transform(X_test) 

    return X_train_trf, X_test_trf,  y_train, y_test, model_input_pipe

@task
def train_model_search(train, valid, y_test):

    def objective(params):

        with mlflow.start_run():
            mlflow.set_tag("developer", "tanmoy")
            mlflow.set_tag("model", "xgboost hyperparam orchestration")
            mlflow.set_tag("type", "experiment")

            mlflow.log_params(params)

            booster = xgb.train(params = params,
                                dtrain = train,
                                num_boost_round = 1000,
                                evals = [(valid, "validation")],
                                early_stopping_rounds = 50)

            
            y_pred = booster.predict(valid).round()
            
            accuracy = accuracy_score(y_test, y_pred)
            mlflow.log_metric("accuracy", accuracy)

            recall = recall_score(y_test, y_pred)
            mlflow.log_metric("recall", recall)

            precision = precision_score(y_test, y_pred)
            mlflow.log_metric("precision", precision)

            f1 = f1_score(y_test, y_pred)
            mlflow.log_metric("f1_score", f1)

            roc_auc = roc_auc_score(y_test, y_pred)
            mlflow.log_metric("roc_auc", roc_auc)

        return {"loss": -recall, 'status': STATUS_OK}

    search_space =  {
    'max_depth' : scope.int(hp.quniform('max_depth', 4, 100, 1)),
    'learning_rate' : hp.loguniform('learning_rate', -3, 0),
    'min_child_weight' : hp.loguniform('min_child_weight', -1, 3),
    'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
    'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
    'objective' : 'binary:logistic',
    'seed' : 786}

    best_result = fmin(
        fn = objective,
        space = search_space,
        algo = tpe.suggest,
        max_evals = 30,
        trials = Trials()
    )

    return

@task
def train_best_model(train, valid, y_test, model_input_pipe):
    
    with mlflow.start_run():

        best_params = {
            'learning_rate'	: 0.04986860974396409,
            'max_depth' :	4,
            'min_child_weight' :	0.04986860974396409,
            'reg_alpha': 0.026933938739370816,
            'reg_lambda': 0.005529647819455234,
            'objective' :	'binary:logistic',
            'seed' :	786
        }

        mlflow.log_params(best_params)
        
        mlflow.set_tag("developer", "tanmoy")
        mlflow.set_tag("model", "xgboost")
        mlflow.set_tag("type", "final")

        xgbooster = xgb.train(
                            params = best_params,
                            dtrain = train,
                            num_boost_round = 100,
                            evals = [(valid, "validation")],
                            early_stopping_rounds = 50)

        y_pred = xgbooster.predict(valid).round()
                
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)

        recall = recall_score(y_test, y_pred)
        mlflow.log_metric("recall", recall)

        precision = precision_score(y_test, y_pred)
        mlflow.log_metric("precision", precision)

        f1 = f1_score(y_test, y_pred)
        mlflow.log_metric("f1_score", f1)

        roc_auc = roc_auc_score(y_test, y_pred)
        mlflow.log_metric("roc_auc_score", roc_auc)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(model_input_pipe, f_out)

        mlflow.log_artifact("models/preprocessor.b", artifact_path = "preprocessor")
        mlflow.xgboost.log_model(xgbooster, artifact_path= "model_mlflow")

    mlflow.end_run()


@flow(task_runner = SequentialTaskRunner())
def main():
    # local_file_system_block = LocalFileSystem.load("mlflow-storage-local")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("persistency-prediction-experiment")
    input_df = read_data(INPUT_FILEPATH, INPUT_FILENAME)
    temp_df = create_features(input_df)
    clean_df = clean_data(temp_df)
    X_train_trf, X_test_trf, y_train, y_test, model_input_pipe = crate_train_test(clean_df)
    train = xgb.DMatrix(X_train_trf, label = y_train)
    valid = xgb.DMatrix(X_test_trf, label = y_test)
    # train_model_search(train, valid, y_test)
    train_best_model(train, valid, y_test, model_input_pipe)

deployment = Deployment.build_from_flow(
    flow=main,
    name="local",
    schedule= IntervalSchedule(interval=timedelta(minutes=5)),
    tags = ["mlflow"]
)

deployment.apply()








