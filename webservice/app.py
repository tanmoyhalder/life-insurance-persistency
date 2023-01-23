import pickle
import os

import pandas as pd
import numpy as np
from datetime import datetime as dt
from datetime import timedelta
import xgboost as xgb

from feature_engine import encoding as ce
from feature_engine import imputation as mdi
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import mlflow

import streamlit as st
st.set_page_config(layout="wide")


LOGGED_MODEL = 'model_config'
LOOKUP_TABLE_PATH = 'lookup_table'
NA_VALUES = ['', 'NA', 'N/A', 'NULL', 'null', '?', '*', '#N/A', '#VALUE!']
POLICY_DTYPE_DICT = {'policy_number': 'str'}
DATE_COLS = ['proposal_received_date', 'policy_issue_date']
NA_VALUES = ['', 'NA', 'N/A', 'NULL', 'null', '?', '*', '#N/A', '#VALUE!']
AGENT_DTYPE_DICT = {'agent_code': 'str'}
ZIPCODE_DTYPE_DICT = {'zipcode': 'str'}
COLS_TO_REM_ONLINE = ['proposal_received_date','policy_issue_date', 'zipcode', 'county', 'state', 'agent_code', 'agent_dob', 'agent_doj']
COLS_TO_REM_BATCH = ['proposal_received_date','policy_issue_date', 'zipcode', 'county', 'state', 'agent_code']
MISSING_COLS = ['agent_persistency']
ONE_HOT_COLS = ['owner_gender', 'marital_status', 'smoker', 'medical', 'education', 'occupation', 'payment_freq',  
                'agent_status', 'agent_education']

LOADING_ERR = KeyError("""The app encountered a problem in initializing the data. 
Try to reload the page. If the problem persists, contact tanmoyhalder123@gmail.com""")

def load_model():

    xgboost_model = mlflow.xgboost.load_model(LOGGED_MODEL)
    with open(os.path.join(LOGGED_MODEL, "preprocessor.b"), "rb") as f_in:
            model_input_pipe = pickle.load(f_in)

    return xgboost_model, model_input_pipe

def load_non_policy_lookup_table():

    agent_tbl = pd.read_csv(os.path.join(LOOKUP_TABLE_PATH, 'agent_data.csv'), dtype = AGENT_DTYPE_DICT)
    zipcode_tbl = pd.read_csv(os.path.join(LOOKUP_TABLE_PATH, 'zipcode_data.csv'), dtype = ZIPCODE_DTYPE_DICT)

    return agent_tbl, zipcode_tbl

def load_policy_lookup_table():

    policy_tbl = pd.read_csv(os.path.join(LOOKUP_TABLE_PATH, 'policy_data.csv'), 
                                          dtype = POLICY_DTYPE_DICT,
                                          na_values = NA_VALUES,
                                          parse_dates = DATE_COLS,
                                          dayfirst= True)

    return policy_tbl


def create_features(df) -> pd.DataFrame:
    df['time_to_issue'] = (df['policy_issue_date'] - df['proposal_received_date']).dt.days
    df['prem_to_income_ratio'] = np.where(df['income'] == 0, 0, (df['annual_premium']/df['income']))

    return df


def clean_data_online(df)  -> pd.DataFrame:
    df = df.drop(COLS_TO_REM_ONLINE, axis = 1)

    return df

def clean_data_batch(df)  -> pd.DataFrame:
    df = df.drop(COLS_TO_REM_BATCH, axis = 1)

    return df


def get_value(val,my_dicts):
	for key,value in my_dicts.items():
		if val == key:
			return value


def get_key(val,my_dicts):
	for key,value in my_dicts.items():
		if val == value:
			return key

def create_final_input_online(preproc_df):

                agent_tbl, zipcode_tbl = load_non_policy_lookup_table()

                model_temp_df1 = preproc_df.merge(agent_tbl, how = 'inner', left_on = 'agent_code', right_on = 'agent_code')
                model_temp_df2 = model_temp_df1.merge(zipcode_tbl, how = 'inner', left_on = 'zipcode', right_on = 'zipcode')

                model_merge_df = create_features(model_temp_df2)
                model_clean_df = clean_data_online(model_merge_df)

                return model_clean_df

def create_final_input_batch(policy_df, policy_tbl):

                policy_tbl = load_policy_lookup_table()

                model_merge_df1 = policy_df.merge(policy_tbl, how = 'inner', left_on = 'policy_number', right_on = 'policy_number').set_index('policy_number')

                model_merge_df2 = create_features(model_merge_df1)
                model_clean_df = clean_data_batch(model_merge_df2)

                return model_clean_df


def main():

    st.title("13th Month Presistency Prediction Web Service")

    html_templ = """
	<div style="background-color:yellow;padding:8px;">
	<h2 style="color:grey">End-to-end ML Application for Life Insurance</h2>
	</div>
	"""

    st.markdown(html_templ,unsafe_allow_html=True)

    activity = ["Online Prediction","Batch Prediction"]
    choice = st.sidebar.selectbox("Choose Service Type", activity)

    if choice == "Online Prediction":

        st.subheader("Prediction Service for a Single Customer")

        col1, padding, col2 = st.columns((10,2,10))

        with col1:

            proposal_received_date = st.date_input("Policy Submission Date", 
                                        help = "Please input the date when the policy was submitted.")

            policy_issue_date = st.date_input("Policy Issue Date", 
                                    help = "Please input the date when the policy was issued/bound.")

            policy_term = st.number_input("Policy Term", min_value = 10, max_value = 30,
                                        help = "Please enter a number between 10 to 30 years.")

            payment_freq = st.selectbox("Premium Payment Frequency",
                                        ('Annually', 'Monthly', 'Quarterly'),
                                        help = "Please select a value from the dropdown.")

            annual_premium = st.number_input("Annual Premium", min_value = 0, max_value = 100000, value = 15000,
                                        help = "Please enter a number greater than 0.")

            sum_insured = st.number_input("Total Sum Insured", min_value = 0, max_value = 10000000, value = 500000,
                                        help = "Please enter a number greater than 0.")

            agent_code = st.text_input("8 Digit Agent Code", value = '60503862',
                                        help = "Please mention the 8 digit code of the agent. Please check agent database for details.")
                                        

        with col2:

            owner_age = st.number_input("Age of the Policy Owner", min_value = 18, max_value = 80, value = 35,
                                        help = "Please enter a number between 18 to 80 years.")

            owner_gender = st.selectbox("Gender of the Policy Owner",
                                    ('Male', 'Female', 'Others'),
                                    help = "Please select a value from the dropdown.")

            marital_status = st.selectbox("Marital Status of the Policy Owner",
                                    ('Single', 'Married', 'Divorced', 'Widowed'),
                                    help = "Please select a value from the dropdown.")

            education = st.selectbox("Education Background of the Policy Owner",
                                    ('Lt Than High School', 'High School', 'Some College', 'Graduate', 'Post Graduate', 'Others', 'Unknown'), ## Need to make unknown to NA
                                    help = "Please select a value from the dropdown.")  

            occupation = st.selectbox("Occupation of the Policy Owner",
                                    ('Sales','Housewife','Other Service','Military','Teacher','Accountant','Govt Service','Shop Owner','IT Service','Businessman','Lawyer','Construction','Student','Manager','Manufacturing','Agricultural','Professional','Doctor','Other Engineering','Unemployed','Retired'),
                                    help = "Please select a value from the dropdown.")

            experience = st.slider("Work Experience of the Policy Owner (in Years)", min_value = 0, max_value = 60, value = 10,
                                        help = "Please select a number between 0 to 60 years.")

            income = st.number_input("Annual Income of the Policy Owner", min_value = 0, max_value = 1000000000, value = 200000,
                                        help = "Please enter a number.")

            credit_score = st.number_input("Credit Score of the Policy Owner", min_value = 300, max_value = 900, value = 650,
                                        help = "Please enter a number.")

            zip_code = st.text_input("Residence Zipcode of the Policy Owner", value = '76543',
                                        help = "Please enter the 5 digit zipcode.")

            family_member = st.number_input("Number of Family Member the Policy Owner", min_value = 0, max_value = 10,
                                        help = "Please enter a number.")

            existing_num_policy = st.number_input("Existing Number of Policies for the Policy Owner/Household", min_value = 0, max_value = 10,
                                        help = "Please enter a number.")

            num_nominee = st.number_input("Number of Nominees for the  Policy", min_value = 1, max_value = 3,
                                        help = "Please enter a number.")

            is_critical_health = {'Yes': 1, 'No': 0}
            choice_critical_health = st.radio("History of Critical Illness in the Family?",
                                        tuple(is_critical_health.keys()),
                                        help = "Please see updated details in manual for definition of critical illness.")
            has_critical_health_history = get_value(choice_critical_health, is_critical_health)
            
            is_smoker = {'Yes': 1, 'No': 0}
            choice_smoker = st.radio("Is the Policy Owner a Smoker?",
                                        tuple(is_smoker.keys()),
                                        help = "Please see updated details in manual for definition of smoker.")
            smoker = get_value(choice_smoker,is_smoker)        
            
            is_medical = {'Yes': 1, 'No': 0}
            choice_medical = st.radio("Medical Tests Done for the Policy Owner?",
                                         tuple(is_medical.keys()),
                                        help = "Please see details of the required medical tests.")
            medical = get_value(choice_medical,is_medical)

            app_input_dict = {
                
                'proposal_received_date': proposal_received_date,
                'policy_issue_date': policy_issue_date,
                'owner_age': owner_age,
                'owner_gender': owner_gender, 
                'marital_status': marital_status, 
                'num_nominee': num_nominee,
                'smoker': smoker,
                'medical': medical, 
                'education': education, 
                'occupation': occupation,
                'experience': experience, 
                'income': income,
                'credit_score': credit_score,
                'family_member': family_member, 
                'existing_num_policy': existing_num_policy,
                'has_critical_health_history': has_critical_health_history, 
                'policy_term': policy_term, 
                'payment_freq': payment_freq,
                'annual_premium': annual_premium, 
                'sum_insured': sum_insured,
                'zipcode': zip_code,
                'agent_code': agent_code

            }

        preproc_df = pd.DataFrame(app_input_dict, index = [0])


        if st.button("Predict"):
            prediction_label = {'NO. The customer is NOT going to renew the policy': 1, 'YES. The customer is going to renew the policy': 0}

            xgboost_model, model_input_pipe = load_model()
            
            model_clean_df = create_final_input_online(preproc_df)
            model_trf_df = model_input_pipe.transform(model_clean_df)
            model_input = xgb.DMatrix(model_trf_df)

            predicted_prob = xgboost_model.predict(model_input)

            if predicted_prob >= 0.5:
                prediction = 1
            else:
                prediction = 0

            final_result = get_key(prediction, prediction_label)
            if prediction == 1:
                st.error(final_result)
                st.error(f"The probability of the customer not paying the premium is {round(predicted_prob[0]*100,0)}%")
            else:
                st.success(final_result) 
                st.success(f"The probability of the customer not paying the premium is {round(predicted_prob[0]*100,0)}%") 
        else:
            pass

    else:

        st.subheader("Prediction Service for a Batch of Customers")

        file = st.file_uploader("Please Upload a File with Policy Numbers   (The algorithm will fetch details of all customers in this list from the database)",
                                help = "Please see here for the required file format required in policy batch file upload.")

        if st.button("Predict"):

            xgboost_model, model_input_pipe = load_model()
            policy_df = pd.read_csv(file, dtype = POLICY_DTYPE_DICT)

            policy_tbl = load_policy_lookup_table()
            model_clean_df = create_final_input_batch(policy_df, policy_tbl)
            model_trf_df = model_input_pipe.transform(model_clean_df)
            model_input = xgb.DMatrix(model_trf_df)

            predicted_prob = np.round(xgboost_model.predict(model_input), 2)

            predicted_prob_df = pd.DataFrame(predicted_prob, columns = ['probabilities'])

            output_df = pd.concat([policy_df, predicted_prob_df], axis = 1)
            output_df['policy_renewal_prediction'] = np.where(output_df['probabilities'] >= 0.5, "No", "Yes")

            st.write(output_df)

            # timestamp = pd.Timestamp.today()
            st.download_button(label = 'Download CSV', data = output_df.to_csv(), mime = 'text/csv')

        else:
            pass
        
                      
if __name__ == "__main__":
    main()


