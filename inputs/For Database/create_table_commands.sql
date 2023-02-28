-- customers 

CREATE TABLE customers (
	customer_id VARCHAR (50) PRIMARY KEY,
	owner_age INTEGER NOT NULL,
	owner_gender VARCHAR (50) NOT NULL,
	marital_status VARCHAR (50) NOT NULL,
	smoker VARCHAR (50) NOT NULL,
	medical VARCHAR (50) NOT NULL,
	education VARCHAR (50) NULL,
	occupation VARCHAR (50) NULL,
	experience INTEGER NULL,
	income INTEGER NOT NULL,
	zipcode VARCHAR (50) NOT NULL,
	county VARCHAR (50) NOT NULL,
	state VARCHAR (50) NOT NULL,
	family_member INTEGER NULL,
	existing_num_policy INTEGER NULL,
	has_critical_health_history VARCHAR (50) NOT NULL,
	credit_score INTEGER NULL,
	has_contacted_in_last_6_months VARCHAR (50) NOT NULL);


-- policy

CREATE TABLE policy(
	policy_number VARCHAR (50) PRIMARY KEY,
	customer_id VARCHAR (50) NOT NULL,
	agent_code VARCHAR (50) NOT NULL,
	proposal_received_date DATE NOT NULL,
	policy_issue_date DATE NOT NULL,
	policy_term INTEGER NOT NULL,
	payment_freq VARCHAR (50) NOT NULL,
	annual_premium INTEGER NOT NULL,
	sum_insured INTEGER NOT NULL,
	num_nominee INTEGER NOT NULL);
	
-- agent

CREATE TABLE agent(

	agent_code VARCHAR (50) PRIMARY KEY,
	agent_status VARCHAR (50) NOT NULL,
	agent_education VARCHAR (50) NULL,
	agent_age INTEGER NOT NULL,
	agent_tenure_days INTEGER NOT NULL,
	agent_persistency FLOAT(3,2) NOT NULL,
	last_6_month_submissions INTEGER NOT NULL,
	average_premium INTEGER NOT NULL,
	is_reinstated VARCHAR (50) NOT NULL,
	prev_persistency FLOAT(3,2) NOT NULL,
	num_complaints INTEGER NOT NULL,
	target_completion_perc FLOAT(3,2) NOT NULL);

-- zipcode

CREATE TABLE zipcode(
	zipcode VARCHAR (50) PRIMARY KEY,
	county VARCHAR (50) NOT NULL,
	state VARCHAR (50) NOT NULL,
	negative_zipcode VARCHAR(50) NOT NULL);
	
-- lapse

CREATE TABLE lapse(
	policy_number INTEGER PRIMARY KEY,	
	lapse INTEGER NOT NULL);



