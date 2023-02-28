import streamlit as st
from utils import (
    load_model,
    load_policy_lookup_table,
    create_features,
    clean_data_batch,
    clean_data_online,
    fetch_data,
)
import plotly.figure_factory as ff
import plotly.graph_objects as go

st.set_page_config(layout="wide")


def load_data_model_insights(df_in):
    df = df_in.copy()
    df["zipcode"] = df["zipcode"].values.astype("str")
    df["agent_code"] = df["agent_code"].values.astype("str")

    df_new = create_features(df)
    model_clean_df = clean_data_online(df_new)
    # model_clean_df = model_clean_df.set_index('policy_number')

    return model_clean_df


st.header("Track How the ML Model and the Data Behaves Over Time")
month_choices = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
months = st.multiselect("Select a month", month_choices, "August")

if len(months):
    st.subheader("Data Drift")

    # policy_tbl = load_policy_lookup_table()
    policy_tbl = fetch_data().drop("lapse", axis=1)
    pred_month = months
    compare_df = (
        policy_tbl.query("@policy_tbl.policy_issue_date.dt.month_name() == @pred_month")
        .reset_index(drop=True)
        .copy()
    )

    # compare_df = policy_tbl.sample(300, replace= False)###### replace with month from dropdown
    master_feature_ls = list(policy_tbl.columns)
    master_feature_ls = [ele for ele in master_feature_ls if ele not in ('customer_id', 'policy_number', 'agent_code')]

    feature = st.selectbox("Select a feature to compare", master_feature_ls)
    st.write("Total number of records in the selected time period :", len(compare_df))
    with st.spinner("Loading...."):
        if policy_tbl[feature].dtype != "O":
            x1 = policy_tbl[feature]
            x2 = compare_df[feature]

            hist_data = [x1, x2]

            group_labels = ["Training Data", "New Data"]
            colors = ["#FFE600", "#CCCCCC"]

            fig = ff.create_distplot(
                hist_data, group_labels, show_hist=False, colors=colors
            )
            fig.update_layout(
                title_text=f"Comparison of {feature.upper()} across Traning Data and New Data"
            )
            st.plotly_chart(fig, use_container_width=True)

        elif policy_tbl[feature].dtype == "O":
            x3 = policy_tbl[feature]
            x4 = compare_df[feature]

            col1, padding, col2 = st.columns((10, 2, 10))

            with col1:
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=x3, marker_color="#FFE600"))
                fig.update_xaxes(type="category", categoryorder="category ascending")
                fig.update_layout(
                    title_text=f"Distribution of {feature.upper()} in Traning Data"
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=x4, marker_color="#CCCCCC"))
                fig.update_xaxes(type="category", categoryorder="category ascending")
                fig.update_layout(
                    title_text=f"Distribution of {feature.upper()} in New Data"
                )

                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Model Drift")

        # col1, padding, col2 = st.columns((10, 2, 10))

        # with col1:
        xgboost_model, model_input_pipe = load_model()

        ## Traning Data
        model_clean_df_train = load_data_model_insights(policy_tbl)
        model_trf_df_train = model_input_pipe.transform(model_clean_df_train)
        model_input_train = model_trf_df_train.copy()

        prediction_train = xgboost_model.predict(model_input_train)
        predicted_prob_train = xgboost_model.predict_proba(model_input_train).astype("float")

        ## New Data
        model_clean_df_new = load_data_model_insights(compare_df)
        model_trf_df_new = model_input_pipe.transform(model_clean_df_new)
        model_input_new = model_trf_df_new.copy()

        prediction_new = xgboost_model.predict(model_input_new)
        predicted_prob_new = xgboost_model.predict_proba(model_input_new).astype("float")


        x1 = predicted_prob_train[:, 1]
        x2 = predicted_prob_new[:, 1]

        hist_data = [x1, x2]

        group_labels = ["Prob of Lapse in Traning Data", "Prob of Lapse in New Data"]
        colors = ["#FFE600", "#CCCCCC"]

        fig = ff.create_distplot(
            hist_data, group_labels, show_hist=False, colors=colors, show_rug=False
        )
        fig.update_layout(
            title_text=f"Comparison of Probability Distribution of Lapse across Training Data & New Data"
        )
        st.plotly_chart(fig, use_container_width=True)

        # with col2:
        #     xgboost_model, model_input_pipe = load_model()

        #     model_clean_df = load_data_model_insights(compare_df)
        #     model_trf_df = model_input_pipe.transform(model_clean_df)
        #     # model_input = xgb.DMatrix(model_trf_df)
        #     model_input = model_trf_df.copy()

        #     prediction = xgboost_model.predict(model_input)
        #     predicted_prob = xgboost_model.predict_proba(model_input).astype("float")

        #     x1 = predicted_prob[:, 0]
        #     x2 = predicted_prob[:, 1]

        #     hist_data = [x1, x2]

        #     group_labels = ["Prob of Renewal", "Prob of Lapse"]
        #     colors = ["#FFE600", "#CCCCCC"]

        #     fig = ff.create_distplot(
        #         hist_data, group_labels, show_hist=False, colors=colors, show_rug=False
        #     )
        #     fig.update_layout(
        #         title_text=f"Probability Distribution of Renewal and Lapse in New Data"
        #     )
        #     st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("Please select a month to continue")
