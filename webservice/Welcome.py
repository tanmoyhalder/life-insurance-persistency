import streamlit as st


st.set_page_config(layout="wide")

st.title("13th Month Presistency Prediction Web Service")

html_templ = """
	<div style="background-color:yellow;padding:8px;">
	<h2 style="color:grey";>End-to-end ML Application for Life Insurance</h2>
	</div>
	"""

st.markdown(html_templ, unsafe_allow_html=True)

st.sidebar.success("Select an option above")
