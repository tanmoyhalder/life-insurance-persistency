import streamlit as st


st.set_page_config(layout="wide")

st.title("13th Month Lapse Prediction Web API for Business Users")

def click_fn(which):
    st.session_state[which] = not st.session_state[which]

st.write("")
st.write("")
st.subheader("Welcome to the app. Please click on the buttons below to know more about it.")
st.write("You can also navigate to the left sidebar panel to visit respective pages")
st.write("")
st.write("")

if "Description Button Status" not in st.session_state:
    st.session_state["Description Button Status"] = False

st.button(
    "Description of the Application",
    on_click = click_fn,
    kwargs = {"which": "Description Button Status"},
)
if st.session_state["Description Button Status"]:
    
	description_html_templ = """
		<div 
		style= 
		padding: 8px;">
			<body>
				<p style ="color: white";>
					13th Month Persistency is one of the KPIs for every Life Insurance Organisation, because it is directly linked to product and organisation profitability. The 
					organisations take great efforts to ensure their customers renew the policies by paying the annual premium by or before the grace period. Though many 
					organisations have their own business logic and strategies in place to ensure their target KPI numbers are met, a machine learning approach can also help the 
					teams to achieve this with better accuracy and efficiency. 
				</p>
				<p style ="color: white";>
					This is a web application targeted at Persistency and Customer Engagament teams of a Life Insurance Company. This app helps the team to understand whether a customer
					will pay his/her 13th Month renewal premium and what might potentially drive this decision. The output is generated from a machine learning model, which outputs the probability 
					of lapse for every customer.
				</p>	 
				<p style ="color: white";>
					The probability varies from 0 to 1. Values closer to 0 means customer has lesser chances of lapse and higher chances of renewal (Ideal customer), whereas 
					values closer to 1 means higher chances of lapse.
				</p>
				<p style ="color: white";>
					This app gives the business visibility of every customer who is due for renewal and also gives flexibility to create strategy around individual customers.
				</p>
			</body>
		</div>
		"""
	st.markdown(description_html_templ, unsafe_allow_html=True)

else:
	pass


if "Features Button Status" not in st.session_state:
    st.session_state["Features Button Status"] = False

st.button(
    "Features of the Application",
    on_click = click_fn,
    kwargs = {"which": "Features Button Status"},
)
if st.session_state["Features Button Status"]:
	features_html_templ = """
	<div>
		<body>
			<p> The app has quite a few useful features integrated. The users can </p>
			<p style ="color: white";>
				<li> <strong>Check the prediction for a single customer:</strong> This feature helps to understand the ideal combination of variables that ensures a highly 
				convertible customer. This combinations can also be shared with frontline sales for for good quality of new business </li>
			</p>
			<p style ="color: white";>
				<li> <strong>View and Analyse Monthly Due Base:</strong> In this feature, a user can check the list of customers due for premium renewal for a particular month
				along with the predicted probabilities and also can inspect what drives the model output. Also, in this feature the user can check the borderline cases where
				the probabilities are very close to the threshold boundary and can tweak the controllable factors to ensure desirable outcome (Convert to renewal from lapse)
				</li> 
			</p>
			<p style ="color: white";>
				<li> <strong>Model Monitoring:</strong> As the name suggests, this page gives <strong>transparency</strong> to the business users on how the data is fairing
				compared to the base data on which the model was trained. This is called <strong>Data Drift</strong>. The users can also check this by every feature. Apart
				from Data Drift, the user can also check how the model is performing in terms of probability distribution by comparing traning output and that of a selected
				month. This is called <strong>Model Drift</strong>
				</li> 
			</p>
		</body>	
	</div>
	"""
	st.markdown(features_html_templ, unsafe_allow_html=True)
else:
	pass

if "Deployment Button Status" not in st.session_state:
    st.session_state["Deployment Button Status"] = False

st.button(
    "Deployment Details of the Application",
    on_click = click_fn,
    kwargs = {"which": "Deployment Button Status"},
)
if st.session_state["Deployment Button Status"]:
	Deployment_html_templ = """
	<div>
		<body>
			<p style ="color: white";>
				The solution is depolyed completely on Google Cloud Platform using services like 
				<li> Cloud SQl </li>
				<li> Cloud Storage</li>
				<li> Cloud Source Reporistory </li> 
				<li> Cloud Build </li>
				<li> Cloud Run </li>
				<li>Artifact Registry</li>
			</p>
		</body>	
	</div>
	"""
	st.markdown(Deployment_html_templ, unsafe_allow_html=True)
else:
	pass

# if "Contact Button Status" not in st.session_state:
#     st.session_state["Contact Button Status"] = False

# st.button(
#     "Contact Us",
#     on_click = click_fn,
#     kwargs = {"which": "Contact Button Status"},
# )
# if st.session_state["Contact Button Status"]:
# 	Deployment_html_templ = """
# 	<div>
# 		<body>
# 			<p style ="color: white";>
# 				Contact EY GDS D&A Team to report any issue with this app or to know more details 
# 				<li> Aparna Mishra - AI Leader</li>
# 				<li> Tanmoy Halder - Senior Data Scientist</li>
# 				<li> Sharvesh Sekar - Data Scientist</li>
# 			</p>
# 		</body>	
# 	</div>
# 	"""
# 	st.markdown(Deployment_html_templ, unsafe_allow_html=True)


# st.write("")
# st.write("")
# st.write("")
# st.write("")
# st.write("")
# st.write("")
# st.write("")

# st.write("Contact EY GDS D&A Team to report any issue with this app or to know more details")

st.sidebar.success("Select an option above")
