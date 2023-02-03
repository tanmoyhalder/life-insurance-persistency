import pandas as pd
import mlflow
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import shap
from datetime import datetime as dt
import yaml


IMAGE_NAME = "online_pred_" + str(dt.today().date()) + ".jpg"

with open("config.yaml", "r") as f:
    config = yaml.load(f, yaml.FullLoader)


def load_model():
    xgboost_model = mlflow.xgboost.load_model(config["LOGGED_MODEL"])
    with open(os.path.join(config["LOGGED_MODEL"], "preprocessor.b"), "rb") as f_in:
        model_input_pipe = pickle.load(f_in)

    return xgboost_model, model_input_pipe


def load_train_data_attribs():
    feat_df = pd.read_parquet(
        os.path.join(config["MODEL_ATTRIB_PATH"], "feat_df.parquet")
    )

    with open(os.path.join(config["MODEL_ATTRIB_PATH"], "X_train_trf.b"), "rb") as f_in:
        X_train_trf = pickle.load(f_in)

    with open(
        os.path.join(config["MODEL_ATTRIB_PATH"], "xgb_explainer.b"), "rb"
    ) as f_in:
        explainer = pickle.load(f_in)

    return feat_df, X_train_trf, explainer


def load_non_policy_lookup_table():
    agent_tbl = pd.read_csv(
        os.path.join(config["LOOKUP_TABLE_PATH"], "agent_data.csv"),
        dtype=config["AGENT_DTYPE_DICT"],
    )
    zipcode_tbl = pd.read_csv(
        os.path.join(config["LOOKUP_TABLE_PATH"], "zipcode_data.csv"),
        dtype=config["ZIPCODE_DTYPE_DICT"],
    )

    return agent_tbl, zipcode_tbl


def load_policy_lookup_table():
    policy_tbl = pd.read_csv(
        os.path.join(config["LOOKUP_TABLE_PATH"], "policy_data.csv"),
        dtype=config["POLICY_DTYPE_DICT"],
        na_values=config["NA_VALUES"],
        parse_dates=config["DATE_COLS"],
        dayfirst=True,
    )

    return policy_tbl


def create_features(df) -> pd.DataFrame:
    df["time_to_issue"] = (
        df["policy_issue_date"] - df["proposal_received_date"]
    ).dt.days
    df["prem_to_income_ratio"] = np.where(
        df["income"] == 0, 0, (df["annual_premium"] / df["income"])
    )

    return df


def clean_data_online(df) -> pd.DataFrame:
    df = df.drop(config["COLS_TO_REM_ONLINE"], axis=1)

    return df


def clean_data_batch(df) -> pd.DataFrame:
    df = df.drop(config["COLS_TO_REM_BATCH"], axis=1)

    return df


def get_value(val, my_dicts):
    for key, value in my_dicts.items():
        if val == key:
            return value


def get_key(val, my_dicts):
    for key, value in my_dicts.items():
        if val == value:
            return key


def create_final_input_online(preproc_df):
    agent_tbl, zipcode_tbl = load_non_policy_lookup_table()

    model_temp_df1 = preproc_df.merge(
        agent_tbl, how="inner", left_on="agent_code", right_on="agent_code"
    )
    model_temp_df2 = model_temp_df1.merge(
        zipcode_tbl, how="inner", left_on="zipcode", right_on="zipcode"
    )

    model_merge_df = create_features(model_temp_df2)
    model_clean_df = clean_data_online(model_merge_df)

    return model_clean_df


def create_final_input_batch(policy_df, policy_tbl):
    policy_tbl = load_policy_lookup_table()

    model_merge_df1 = policy_df.merge(
        policy_tbl, how="inner", left_on="policy_number", right_on="policy_number"
    ).set_index("policy_number")

    model_merge_df2 = create_features(model_merge_df1)
    model_clean_df = clean_data_batch(model_merge_df2)

    return model_clean_df


def create_image_df(model_input_pipe, model_input):
    model_final_features = model_input_pipe.get_feature_names_out(
        input_features=config["FEATURES"]
    )

    out_feature_list = []
    for f in range(0, len(model_final_features)):
        feat = "f" + str(f)
        out_feature_list.append(feat)

    feat_df = pd.DataFrame(
        data=model_final_features, index=out_feature_list, columns=["feature_names"]
    )
    # feat_df
    plot_df = pd.DataFrame(model_input, columns=feat_df["feature_names"].to_list())
    return plot_df


def create_save_shap_plot(explainer, plot_df, path, image_name):
    shap_values = explainer.shap_values(plot_df)
    expected_value = explainer.expected_value

    explainer_img = shap.plots._waterfall.waterfall_legacy(
        expected_value,
        shap_values[0],
        features=plot_df.squeeze(),
        feature_names=plot_df.columns,
        max_display=15,
        show=False,
    )

    plt.tight_layout()
    explainer_img.savefig(os.path.join(path, image_name))
