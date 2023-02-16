import streamlit as st
import numpy as np
from millify import millify
import pandas as pd
from utils import load_df_for_plots, plot_map, descriptive_polts


st.set_page_config(layout="wide")

df = load_df_for_plots()
cust_cnt = len(df)
policy_cnt = len(df.index)
lapse_rate = round((df["lapse"].value_counts() / len(df))[1], 3)
num_agents = df["agent_code"].nunique()
tot_annual_prem = df["annual_premium"].sum()
tot_sum_assured = df["sum_insured"].sum()
avg_policy_tenure = round(df["policy_term"].mean())
avg_tkt_size = tot_annual_prem / cust_cnt
revenue_leakage = tot_annual_prem * lapse_rate
agg_df = (
    df.groupby("state")
    .agg(
        {
            "policy_issue_date": pd.Series.count,
            "annual_premium": (lambda x: np.round(pd.Series.mean(x), 1)),
            "lapse": np.sum,
            "agent_code": pd.Series.count,
        }
    )
    .reset_index()
)
agg_df = agg_df.rename(
    {
        "policy_issue_date": "Policy Count",
        "annual_premium": "Annual Premium",
        "lapse": "Lapse",
        "agent_code": "Agent Count",
    },
    axis=1,
)

col1, col2, col3, col4 = st.columns([4, 1, 1, 1])

with col1:
    st.subheader("Values across US States")
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Policy Count", "Annual Premium", "Lapse Count", "Agent Count"]
    )

    with tab1:
        fig = plot_map(agg_df, "Policy Count")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = plot_map(agg_df, "Annual Premium")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = plot_map(agg_df, "Lapse")
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        fig = plot_map(agg_df, "Agent Count")
        st.plotly_chart(fig, use_container_width=True)

with col3:
    st.subheader("Key Indicators")
    st.metric("Customer Count", f"{cust_cnt:,}")
    st.metric("Policy Count", f"{policy_cnt:,}")
    st.metric("Lapse Rate", f"{lapse_rate*100}%")
    st.metric("Number of agents", f"{num_agents:,}")
    st.metric("Average Policy Tenure", f"{avg_policy_tenure} years")

with col4:
    st.subheader("⠀⠀⠀⠀⠀")
    st.metric("Total Annual Premium", f"${millify(tot_annual_prem, precision=1)}")
    st.metric("Total Sum Insured", f"${millify(tot_sum_assured, 1)}")
    st.metric("Average Ticket Size", f"${round(avg_tkt_size):,}")
    st.metric("Revenue Leakage", f"${millify(revenue_leakage,1)}")

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
        "Lapse",
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

    x_axis = st.selectbox(label="View distribution of", options=x_vars, index=0)
    group_var = st.selectbox(label="Drill-down by", options=group_vars, index=0)

    fig = descriptive_polts(df, x_axis, group_var)

    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, theme=None)
    else:
        st.write("Invalid Combination")

with sub_tab2:
    x_vars = ["Policy Term", "Payment Frequency", "Annual Premium", "Sum Insured"]
    group_vars = [
        "Lapse",
        "Policy Term",
        "Payment Frequency",
        "Annual Premium",
        "Sum Insured",
    ]

    x_axis = st.selectbox(label="View distribution of", options=x_vars, index=0)
    group_var = st.selectbox(label="Drill-down by", options=group_vars, index=0)

    fig = descriptive_polts(df, x_axis, group_var)

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
        "Lapse",
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

    fig = descriptive_polts(df, x_axis, group_var)

    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, theme=None)
    else:
        st.write("Invalid Combination")
