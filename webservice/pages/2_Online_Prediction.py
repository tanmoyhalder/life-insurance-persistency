import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
from utils import (
    load_model,
    create_final_input_online,
    get_key,
    load_train_data_attribs,
    create_image_df,
    create_save_shap_plot,
    get_value,
)
from utils import IMAGE_NAME, config
import os

st.subheader("Prediction Service for a Single Customer")

col1, padding, col2 = st.columns((10, 2, 10))

with col1:
    proposal_received_date = st.date_input(
        "Policy Submission Date",
        help="Please input the date when the policy was submitted.",
    )

    policy_issue_date = st.date_input(
        "Policy Issue Date",
        help="Please input the date when the policy was issued/bound.",
    )

    policy_term = st.number_input(
        "Policy Term",
        min_value=10,
        max_value=30,
        help="Please enter a number between 10 to 30 years.",
    )

    payment_freq = st.selectbox(
        "Premium Payment Frequency",
        ("Annually", "Monthly", "Quarterly"),
        help="Please select a value from the dropdown.",
    )

    annual_premium = st.number_input(
        "Annual Premium",
        min_value=0,
        max_value=100000,
        value=15000,
        help="Please enter a number greater than 0.",
    )

    sum_insured = st.number_input(
        "Total Sum Insured",
        min_value=0,
        max_value=10000000,
        value=500000,
        help="Please enter a number greater than 0.",
    )

    agent_code = st.text_input(
        "8 Digit Agent Code",
        value="60503862",
        help="Please mention the 8 digit code of the agent. Please check agent database for details.",
    )

with col2:
    owner_age = st.number_input(
        "Age of the Policy Owner",
        min_value=18,
        max_value=80,
        value=35,
        help="Please enter a number between 18 to 80 years.",
    )

    owner_gender = st.selectbox(
        "Gender of the Policy Owner",
        ("Male", "Female", "Others"),
        help="Please select a value from the dropdown.",
    )

    marital_status = st.selectbox(
        "Marital Status of the Policy Owner",
        ("Single", "Married", "Divorced", "Widowed"),
        help="Please select a value from the dropdown.",
    )

    education = st.selectbox(
        "Education Background of the Policy Owner",
        (
            "Lt Than High School",
            "High School",
            "Some College",
            "Graduate",
            "Post Graduate",
            "Others",
            "Unknown",
        ),  ## Need to make unknown to NA
        help="Please select a value from the dropdown.",
    )

    occupation = st.selectbox(
        "Occupation of the Policy Owner",
        (
            "Sales",
            "Housewife",
            "Other Service",
            "Military",
            "Teacher",
            "Accountant",
            "Govt Service",
            "Shop Owner",
            "IT Service",
            "Businessman",
            "Lawyer",
            "Construction",
            "Student",
            "Manager",
            "Manufacturing",
            "Agricultural",
            "Professional",
            "Doctor",
            "Other Engineering",
            "Unemployed",
            "Retired",
        ),
        help="Please select a value from the dropdown.",
    )

    experience = st.slider(
        "Work Experience of the Policy Owner (in Years)",
        min_value=0,
        max_value=60,
        value=10,
        help="Please select a number between 0 to 60 years.",
    )

    income = st.number_input(
        "Annual Income of the Policy Owner",
        min_value=0,
        max_value=1000000000,
        value=200000,
        help="Please enter a number.",
    )

    credit_score = st.number_input(
        "Credit Score of the Policy Owner",
        min_value=300,
        max_value=900,
        value=650,
        help="Please enter a number.",
    )

    zip_code = st.text_input(
        "Residence Zipcode of the Policy Owner",
        value="76543",
        help="Please enter the 5 digit zipcode.",
    )

    family_member = st.number_input(
        "Number of Family Member the Policy Owner",
        min_value=0,
        max_value=10,
        help="Please enter a number.",
    )

    existing_num_policy = st.number_input(
        "Existing Number of Policies for the Policy Owner/Household",
        min_value=0,
        max_value=10,
        help="Please enter a number.",
    )

    num_nominee = st.number_input(
        "Number of Nominees for the  Policy",
        min_value=1,
        max_value=3,
        help="Please enter a number.",
    )

    is_critical_health = {"Yes": 1, "No": 0}
    choice_critical_health = st.radio(
        "History of Critical Illness in the Family?",
        tuple(is_critical_health.keys()),
        help="Please see updated details in manual for definition of critical illness.",
    )
    has_critical_health_history = get_value(choice_critical_health, is_critical_health)

    is_smoker = {"Yes": 1, "No": 0}
    choice_smoker = st.radio(
        "Is the Policy Owner a Smoker?",
        tuple(is_smoker.keys()),
        help="Please see updated details in manual for definition of smoker.",
    )
    smoker = get_value(choice_smoker, is_smoker)

    is_medical = {"Yes": 1, "No": 0}
    choice_medical = st.radio(
        "Medical Tests Done for the Policy Owner?",
        tuple(is_medical.keys()),
        help="Please see details of the required medical tests.",
    )
    medical = get_value(choice_medical, is_medical)

    app_input_dict = {
        "proposal_received_date": proposal_received_date,
        "policy_issue_date": policy_issue_date,
        "owner_age": owner_age,
        "owner_gender": owner_gender,
        "marital_status": marital_status,
        "num_nominee": num_nominee,
        "smoker": smoker,
        "medical": medical,
        "education": education,
        "occupation": occupation,
        "experience": experience,
        "income": income,
        "credit_score": credit_score,
        "family_member": family_member,
        "existing_num_policy": existing_num_policy,
        "has_critical_health_history": has_critical_health_history,
        "policy_term": policy_term,
        "payment_freq": payment_freq,
        "annual_premium": annual_premium,
        "sum_insured": sum_insured,
        "zipcode": zip_code,
        "agent_code": agent_code,
    }

preproc_df = pd.DataFrame(app_input_dict, index=[0])

online_button_col1, online_button_col2 = st.columns([1, 1])

with online_button_col1:
    if st.button("Predict for this customer "):
        prediction_label = {
            "NO. The customer is NOT going to renew the policy": 1,
            "YES. The customer is going to renew the policy": 0,
        }

        xgboost_model, model_input_pipe = load_model()

        model_clean_df = create_final_input_online(preproc_df)
        model_trf_df = model_input_pipe.transform(model_clean_df)
        # model_input = xgb.DMatrix(model_trf_df)
        model_input = model_trf_df.copy()

        prediction = xgboost_model.predict(model_input)
        predicted_prob = xgboost_model.predict_proba(model_input).astype("float")

        # if predicted_prob >= 0.5:
        #     prediction = 1
        # else:
        #     prediction = 0

        final_result = get_key(prediction, prediction_label)
        if prediction == 1:
            st.error(final_result)
            if (predicted_prob[0, 1] > 0.5) and (predicted_prob[0, 1] < 0.7):
                st.info(
                    f"The probability of the customer not paying the premium is **{np.round(predicted_prob[0,1]*100,0)}%**. There is a low confidence in the output. Please see the explanation below !!!"
                )
            elif predicted_prob[0, 1] >= 0.7:
                st.info(
                    f"The probability of the customer not paying the premium is **{np.round(predicted_prob[0,1]*100,0)}%**. There is a high confidence in the output. Please see the explanation below !!!"
                )

        else:
            st.success(final_result)
            if (predicted_prob[0, 0] > 0.5) and (predicted_prob[0, 0] < 0.7):
                st.info(
                    f"The probability of the customer paying the premium is **{np.round(predicted_prob[0,0]*100,0)}%**. There is a low confidence in the output. Please see the explanation below !!!"
                )
            elif predicted_prob[0, 0] >= 0.7:
                st.info(
                    f"The probability of the customer paying the premium is **{np.round(predicted_prob[0,0]*100,0)}%**. There is a high confidence in the output. Please see the explanation below !!!"
                )

with online_button_col2:
    if st.button("Explain the result"):
        with st.spinner("Calculating..."):
            feat_df, X_train_trf, explainer = load_train_data_attribs()
            xgboost_model, model_input_pipe = load_model()
            model_clean_df = create_final_input_online(preproc_df)
            model_trf_df = model_input_pipe.transform(model_clean_df)
            model_input = model_trf_df.copy()

            plot_df = create_image_df(model_input_pipe, model_input)

            create_save_shap_plot(
                explainer,
                plot_df,
                path=config["IMAGE_PATH"],
                image_name=IMAGE_NAME,
            )

            image = Image.open(os.path.join(config["IMAGE_PATH"], IMAGE_NAME))
            st.image(
                image,
                caption="Shapley Waterfall chart for the customer explaining the factors impacting the decision",
                use_column_width=False,
            )
