import streamlit as st
import pandas as pd
from PIL import Image
import os
import numpy as np
from utils import config, IMAGE_NAME
from utils import (
    load_model,
    load_policy_lookup_table,
    create_final_input_batch,
    load_train_data_attribs,
    create_image_df,
    create_save_shap_plot,
)


st.subheader("Prediction Service for a Batch of Customers")

file = st.file_uploader(
    "Please Upload a File with Policy Numbers   (The algorithm will fetch details of all customers in this list from the database)",
    help="Please see here for the required file format required in policy batch file upload.",
)

if file is not None:
    policy_df = pd.read_csv(file, dtype=config["POLICY_DTYPE_DICT"])
    # if file is None:
    #     st.warning('Please upload a file')
    # else:
    #     policy_df = pd.read_csv(file, dtype = POLICY_DTYPE_DICT)
    policy_id_tp = tuple(list(policy_df["policy_number"].drop_duplicates().values))

    if st.button("Predict for all the customers"):
        xgboost_model, model_input_pipe = load_model()

        if file is not None:
            file.seek(0)
            policy_df = pd.read_csv(file, dtype=config["POLICY_DTYPE_DICT"])
        policy_id_tp = tuple(list(policy_df["policy_number"].drop_duplicates().values))

        policy_tbl = load_policy_lookup_table()
        model_clean_df = create_final_input_batch(policy_df, policy_tbl)
        model_trf_df = model_input_pipe.transform(model_clean_df)
        model_input = model_trf_df.copy()

        predicted_prob = xgboost_model.predict_proba(model_input)[:, 1]

        predicted_prob_df = pd.DataFrame(
            predicted_prob, columns=["probability of lapse"]
        )
        output_df = pd.concat([policy_df, predicted_prob_df], axis=1)
        output_df["model_decision"] = np.where(
            output_df["probability of lapse"] > 0.5, "No Renewal", "Renewal"
        )

        st.download_button(
            label="Download Result CSV",
            data=output_df.to_csv(),
            mime="text/csv",
        )

    selected_policy = st.selectbox(
        "Select a Policy Number for Explanation",
        policy_id_tp,
        help="Please select a Policy Number from the dropdown.",
    )

    batch_button_col1, batch_button_col2 = st.columns([1, 1])

    with batch_button_col1:
        if st.button("Explain the result for the selected policy"):
            with st.spinner("Calculating..."):
                feat_df, X_train_trf, explainer = load_train_data_attribs()
                xgboost_model, model_input_pipe = load_model()
                policy_tbl = load_policy_lookup_table()
                policy_selected_df = policy_df.loc[
                    policy_df["policy_number"] == selected_policy
                ]
                model_clean_df = create_final_input_batch(
                    policy_selected_df, policy_tbl
                )
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

    with batch_button_col2:
        if st.button("Show Values for the selected policy"):
            policy_tbl = load_policy_lookup_table()
            model_clean_df = create_final_input_batch(policy_df, policy_tbl)
            model_clean_df = model_clean_df.reset_index()
            model_clean_df = model_clean_df.rename(columns={0: "policy_number"})
            output_df = model_clean_df.loc[
                model_clean_df["policy_number"] == selected_policy
            ]
            st.dataframe(output_df.squeeze())
else:
    st.warning("Please upload a file with policy numbers !!!")
