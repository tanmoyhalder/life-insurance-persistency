#### Install prefect
`pip install prefect`
#or
`conda install -c conda-forge prefect`

#### Execute a prefect flow one time
`python model_pipeline_orchestration.py`

#### Launch prefect ui
`prefect orion start`

#### Deployment 

For deployment of prefect, we need to use a storage or "block" for storing our artifacts like flows and tasks. This can be done both using a cloud storage or a local file system. I will use a local file system. Using he prefect UI to create the local file storage doesn't work. See here

https://github.com/PrefectHQ/prefect/issues/6082

The below command works

- `prefect deployment build prefect_deployment.py:main --name local --tag mlflow` Note: The name andtag has to be the same as mentioned in your code. See **prefect_deployment.py**. Also this will create an YAML file in your working directory name `main-deployment.yaml`
- `prefect deployment apply main-deployment.yaml` ## applies the configurations in the yaml file
- `prefect deployment run main/local` ## Intitates the runs
- `prefect agent start --tag mlflow` ## Starts an agent which will look for work i.e execute the runs

source: https://github.com/anna-geller/prefect-deployment-patterns/blob/2530e57cc1769f4cdf8476ee795b762068fafb94/a_project_template/deploy_commands_local_fs.bash#L2

I have set up an interval schedule of every 5 minutes. This will create a deployment flow. The agent will look for the run and every 5 minute executethe same code. This is how we automate the workflow
