from matplotlib import pyplot as plt
import streamlit as st
import numpy as np
from utils import clean_data_batch, descriptive_polts
from utils import (
    load_model,
    create_final_input_batch,
    load_train_data_attribs,
    create_image_df,
    create_shap_plot,
    load_df_for_plots,
    fetch_data,
    fetch_column_distict,
)
from millify import millify
import pandas as pd


def click_fn(which):
    st.session_state[which] = not st.session_state[which]


st.set_page_config(layout="wide")

st.header("Prediction for Monthly Due Base")
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
month = st.selectbox("Select a month", month_choices)

# threshold = st.number_input('Lapse Probability Threshold', 0.0, 1.0, 0.5, step=0.01)
threshold = 0.5
classes = ["Renewed", "Lapsed"]

# df = load_df_for_plots()
# df = df.drop(["lapse"], axis=1)

pred_month = month_choices[month_choices.index(month) - 1]
# result_df = (
#     df.query("@df.policy_issue_date.dt.month_name() == @pred_month")
#     .reset_index(drop=True)
#     .copy()
# )
result_df = fetch_data(month_name=pred_month)

xgboost_model, model_input_pipe = load_model()
batch_df = create_final_input_batch(result_df).dropna()
input_data = model_input_pipe.transform(batch_df.drop("customer_id", axis=1))

predicted_prob = xgboost_model.predict_proba(input_data)[:, 1]
batch_df["pred_prob"] = predicted_prob
batch_df["pred_lapse"] = np.where(predicted_prob < threshold, 0, 1)

col1, col2 = st.columns(2)
with col1:
    # 1
    st.metric("Customer Count for this Month", len(predicted_prob))
    # 3
    lapse_counts = batch_df["pred_lapse"].value_counts().tolist()
    if len(predicted_prob) > 1:
        lapse_percentage = lapse_counts[1] / sum(lapse_counts)
        st.metric("Predicted Lapse Percentage", f"{round(100*lapse_percentage, 1)}%")

with col2:
    # 2
    total_premium = batch_df.annual_premium.sum()
    st.metric("Total Due Premium", f"${millify(total_premium, 1)}")
    # 4
    if len(predicted_prob) > 1:
        st.metric(
            "Potential Revenue Leakage (Due to Lapse)",
            f"${millify(lapse_percentage*total_premium, 1)}",
        )

st.write("-------")

####
st.subheader("Descriptive Plots")

sub_tab1, sub_tab2, sub_tab3 = st.tabs(
    ["Customer Attributes", "Policy Attributes", "Agent Attributes"]
)

with sub_tab1:
    x_vars = [
        "Occupation",
        "Smoker",
        "Marital Status",
        "Gender",
        "Education",
        "Experience",
        "Income",
        "Age",
        "Credit Score",
        "Number of Nominees",
        "Underwent Medical Test?",
        "Number of Family Members",
        "Number Existing Policies",
        "Has Critical Health History?",
    ]
    group_vars = [
        "None",
        "Pred Lapse",
        "Occupation",
        "Smoker",
        "Marital Status",
        "Gender",
        "Experience",
        "Income",
        "Age",
        "Credit Score",
        "Number of Nominees",
        "Underwent Medical Test?",
        "Number of Family Members",
        "Number Existing Policies",
        "Has Critical Health History?",
    ]
    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
        x_axis = st.selectbox(label="View distribution of", options=x_vars, index=0)
    with sub_col2:
        group_var = st.selectbox(label="Drill-down by", options=group_vars, index=0)

    fig = descriptive_polts(batch_df, x_axis, group_var)

    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, theme=None)
    else:
        st.write("Invalid Combination")

with sub_tab2:
    x_vars = ["Policy Term", "Payment Frequency", "Annual Premium", "Sum Insured"]
    group_vars = [
        "None",
        "Pred Lapse",
        "Policy Term",
        "Payment Frequency",
        "Annual Premium",
        "Sum Insured",
    ]

    x_axis = st.selectbox(label="View distribution of", options=x_vars, index=0)
    group_var = st.selectbox(label="Drill-down by", options=group_vars, index=0)

    fig = descriptive_polts(batch_df, x_axis, group_var)

    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, theme=None)
    else:
        st.write("Invalid Combination")

with sub_tab3:
    x_vars = [
        "Agent Age",
        "Agent Tenure (days)",
        "Persistency",
        "Last 6 months policies",
        "Average Premium",
        "Is Reinstated?",
        "Persistency before Reinstated",
        "Number Complaints",
        "Agent Status",
        "Agent Education",
        "Target Completion Percentage",
    ]
    group_vars = [
        "None",
        "Pred Lapse",
        "Agent Age",
        "Agent Tenure (days)",
        "Persistency",
        "Last 6 months policies",
        "Average Premium",
        "Is Reinstated?",
        "Persistency before Reinstated",
        "Number Complaints",
        "Agent Status",
        "Agent Education",
        "Target Completion Percentage",
    ]

    x_axis = st.selectbox(label="View distribution of", options=x_vars, index=0)
    group_var = st.selectbox(label="Drill-down by", options=group_vars, index=0)

    fig = descriptive_polts(batch_df, x_axis, group_var)

    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, theme=None)
    else:
        st.write("Invalid Combination")

st.write("-------")
#####################
st.subheader("Model Prediction Explanation")
customer = st.selectbox(
    "Select a customer from below",
    batch_df["customer_id"],
)

customer_datapoint = batch_df.query("@batch_df.customer_id == @customer")

if customer_datapoint.pred_lapse.item() == 1:
    st.error("Prediction : Lapsed")
else:
    st.success("Prediction : Renewal")

if "Waterfall Button Status" not in st.session_state:
    st.session_state["Waterfall Button Status"] = False

st.button(
    "Explain the prediction",
    on_click=click_fn,
    kwargs={"which": "Waterfall Button Status"},
)
if st.session_state["Waterfall Button Status"]:
    feat_df, X_train_trf, explainer = load_train_data_attribs()
    customer_datapoint = clean_data_batch(customer_datapoint).dropna()
    transformed_datapoint = model_input_pipe.transform(
        customer_datapoint.drop("customer_id", axis=1)
    )
    plot_df = create_image_df(model_input_pipe, transformed_datapoint)
    plt.clf()

    fig = create_shap_plot(
        explainer,
        plot_df,
    )
    st.pyplot(fig)
    st.caption(
        "Shapley Waterfall chart for the customer explaining the factors impacting the decision"
    )

st.write("--------")

######## Borderline Analysis

st.subheader("Borderline Customer Analysis (What-If Tool)")
st.markdown(
    'Borderline Cases are the customers where the probability is very close to the decision boundary. Hence, business should take a closer look into these customers for further investigations.  \n \
        We have created a tool where users can experiment with various controllers/ drivers to change the model decision. Below is an example usage of the tool:  \n  \n For a customer, if the original model prediction is "Lapse", then the business might be interested in knowing what they can do to make the customer convert or "Renew" the premium. \
        In simpler terms, what strategy the business can use to ensure higher chances of renewal.  \n\
        There are a few levers, which are under the control of the business like agent performance, customer engagement etc. which can influence the customer decision.\n\
        Since, not all customers will be of equal priority, we have arranged these borderline customer is descending order of their annual premium. ',
    unsafe_allow_html=True,
)
try:
    borderline_datapoints = batch_df.query("0.35 <= pred_prob <= 0.65")

    st.write("")
    st.write("")
    st.write("")
    st.write("")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Customer Count", len(borderline_datapoints))
    with col2:
        st.metric(
            "Total Premium",
            f"${millify(borderline_datapoints.annual_premium.sum(), 2)}",
        )

    highvalue_cutoff = borderline_datapoints.annual_premium.quantile(0.75)
    highvalue_datapoints = borderline_datapoints[
        borderline_datapoints["annual_premium"] > highvalue_cutoff
    ].sort_values("annual_premium", ascending=False)

    with col3:
        st.metric("High Value Customer Count", len(highvalue_datapoints))
    with col4:
        st.metric(
            "High Value Premium",
            f"${millify(highvalue_datapoints.annual_premium.sum(),2)}",
        )

    st.write("")
    st.write("")
    st.write("")

    border_choice = st.selectbox(
        "Choose a Borderline Case",
        borderline_datapoints.sort_values("annual_premium", ascending=False)[
            "customer_id"
        ].unique(),
    )

    original_datapoint = (
        borderline_datapoints.query("customer_id == @border_choice")
        .iloc[0]
        .copy()
        .to_dict()
    )
    col__1, col__2 = st.columns(2)

    with col__1:
        if border_choice in highvalue_datapoints.customer_id.values:
            st.warning(f"High Value Customer")
    with col__2:
        st.info(f"Annual Premium : ${millify(original_datapoint['annual_premium'], 2)}")

    col_1, col_2, col_3 = st.columns(3)

    modified_datapoint = {}

    with col_1:
        st.caption("Customer features")
        # smoker_options = df["smoker"].unique().tolist()
        # medical_options = df["medical"].unique().tolist()
        smoker_options = fetch_column_distict("smoker", "customers")
        medical_options = fetch_column_distict("medical", "customers")
        credit_score_min, credit_score_max = fetch_column_distict(
            "credit_score", "customers"
        )

        modified_datapoint["smoker"] = st.selectbox(
            "Smoker",
            smoker_options,
            index=smoker_options.index(original_datapoint["smoker"]),
        )
        modified_datapoint["medical"] = st.selectbox(
            "Underwent Medical Test?",
            medical_options,
            index=medical_options.index(original_datapoint["medical"]),
        )

        modified_datapoint["credit_score"] = st.slider(
            "Credit Score",
            # df["credit_score"].min().item(),
            # df["credit_score"].max().item(),
            credit_score_min,
            credit_score_max,
            value=original_datapoint["credit_score"],
        )

    with col_2:
        st.caption("Policy features")
        # payment_freq_options = df["payment_freq"].unique().tolist()
        payment_freq_options = fetch_column_distict("payment_freq", "policy")

        modified_datapoint["payment_freq"] = st.selectbox(
            "Payment Frequency",
            payment_freq_options,
            index=payment_freq_options.index(original_datapoint["payment_freq"]),
        )

    with col_3:
        st.caption("Agent features")
        # agent_education_options = df["agent_education"].unique().tolist()
        # contacted_options = df["has_contacted_in_last_6_months"].unique().tolist()
        # agent_status_options = df["agent_status"].unique().tolist()

        agent_education_options = fetch_column_distict("agent_education", "agent")
        contacted_options = fetch_column_distict(
            "has_contacted_in_last_6_months", "customers"
        )
        agent_status_options = fetch_column_distict("agent_status", "agent")
        agent_persistency_min, agent_persistency_max = fetch_column_distict(
            "agent_persistency", "agent"
        )
        target_completion_perc_min, target_completion_perc_max = fetch_column_distict(
            "target_completion_perc", "agent"
        )
        (
            last_6_month_submissions_min,
            last_6_month_submissions_max,
        ) = fetch_column_distict("last_6_month_submissions", "agent")
        average_premium_min, average_premium_max = fetch_column_distict(
            "average_premium", "agent"
        )

        modified_datapoint["agent_status"] = st.selectbox(
            "Agent Status",
            agent_status_options,
            index=agent_status_options.index(original_datapoint["agent_status"]),
        )

        # st.write([int(x) for x in contacted_options])
        modified_datapoint["has_contacted_in_last_6_months"] = int(
            st.selectbox(
                "Has contacted in last 6 months",
                contacted_options,
                index=contacted_options.index(
                    original_datapoint["has_contacted_in_last_6_months"]
                ),
                format_func=lambda x: "Yes" if x == 1 else "No",
            )
        )

        modified_datapoint["agent_persistency"] = st.slider(
            "Last one Year Persistency",
            # df["agent_persistency"].min().item(),
            # df["agent_persistency"].max().item(),
            agent_persistency_min,
            agent_persistency_max,
            value=original_datapoint["agent_persistency"],
        )
        modified_datapoint["target_completion_perc"] = st.slider(
            "Target Completion Percentage",
            # df["target_completion_perc"].min().item(),
            # df["target_completion_perc"].max().item(),
            target_completion_perc_min,
            target_completion_perc_max,
            value=original_datapoint["target_completion_perc"],
        )

        modified_datapoint["last_6_month_submissions"] = st.slider(
            "Last 6 Months Submissions",
            # df["last_6_month_submissions"].min().item(),
            # df["last_6_month_submissions"].max().item(),
            last_6_month_submissions_min,
            last_6_month_submissions_max,
            value=original_datapoint["last_6_month_submissions"],
        )

        modified_datapoint["average_premium"] = st.slider(
            "Average Premium",
            # df["average_premium"].min().item(),
            # df["average_premium"].max().item(),
            average_premium_min,
            average_premium_max,
            value=original_datapoint["average_premium"],
        )

    # average premium, num complaints, target completion percentage, has contacted
    modified_datapoint_df = pd.DataFrame(
        {**original_datapoint, **modified_datapoint}, index=[0]
    )
    customer_datapoint = clean_data_batch(modified_datapoint_df).dropna()
    transformed_customer_datapoint = model_input_pipe.transform(
        customer_datapoint.drop("customer_id", axis=1)
    )

    predicted_prob = xgboost_model.predict_proba(transformed_customer_datapoint)[:, 1]
    prediction_lapse = (predicted_prob > threshold) * 1

    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")

    col__1, _, col__3 = st.columns(3)
    labels = ["Renewal", "Lapsed"]
    with col__1:
        st.metric("Original Prediction", labels[original_datapoint["pred_lapse"]])
        # st.metric("Original Probability", np.round(original_datapoint["pred_prob"], 4))

    with col__3:
        st.metric("New Prediction", labels[int(prediction_lapse)])
        # st.metric(
        #     "New Probability",
        #     np.round(float(predicted_prob), 4),
        #     delta=round(float(predicted_prob) - original_datapoint["pred_prob"], 4),
        #     delta_color="inverse",
        # )
except Exception as e:
    st.info("No Borderline Case for this month")
    # st.error(e)
    print(e)
