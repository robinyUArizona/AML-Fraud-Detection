# Anti Money Laundering (AML) Fraud Detection
##  Life cycle of Machine Learning Project:

            1. Understanding the Problem Statement
            2. Data Collection
            3. Data Checks to perform
            4. Exploratory Data Analysis
            5. Data Pre-Processing
            6. Model Evaluation and Training
            7. Choose Best Model


### Problem Statement
Money laundering is a multi-billion dollar issue. Detection of laundering is very difficult. Most automated algorithms have a high false positive rate: legitimate transactions incorrectly flagged as laundering. The converse is also a major problem -- false negatives, i.e. undetected laundering transactions. 

### Goal
To predict the the given transaction is fraud or not.

### Solution Scope
This can be used in real life by finanicial corporation so that they can improve their AML fraud detection system.

### Solution Approach
1. Machine Learning: ML Classification Algorithms
2. Deep Learning: Custom ANN with sigmoid activation Funtion

### Solution Proposed
1. Download the data from Kaggle
2. Perform EDA and feature engineering to select the desirable features
3. Fit the ML classification algorithm and find out which one performs better
4. Select top few and tune hyperparameters
5. Select the best model based on desired metrics




### Project setup
- Setup the GitHub repository (creating repo and cloning repo locally)
	by creating a new repository on GitHub or by creating a new directory, and initialize it
	```
	echo "# projectname" >> README.md or touch README.md
	git init
	git add README.md
	git commit -m "first commit"
	git branch -M main
	git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY-NAME.git
	git push -u origin main
	```
- Template (Deployable code) using `template.py`
	``` 
	- aml_fraud_detector
		- components
			- data_ingestion.py
			- data_validation.py
			- data_transformation.py
			- model_trainer.py
			- model_evaluation.py
			- model_pusher.py
		- configuration
		- constant
		- entity
			- config_entity.py
			- artifact_entity.py
		- logger
		- pipeline
			- training_pipeline.py
			- prediction_pipeline.py
		- utils
			- main_utils.py
	
	```
- Requirements
	```bash
	conda create -p venv python=3.8 -y
	```
	```bash
	conda activate venv
	```
	```bash
	pip install -r requirements.txt
	Note: -e . at the end in requirements.txt file -> This is for `setup.py` file
	```
- Database setup - MongoDB
	- create project `aml_fraud_detector_mongoDB`
		- create a cluster `cluster-aml-fraud-detector`
		- Setup Network Access
	- Downloaded data from kaggle and insert or upload the data to the MongoDB database (Atlas)

#### Logger and Exceptions setup
- Create directories for logger and exception: The logger and exception will be imported from here throughout the projects
	- aml_fraud_detector
		- logger
			- __init__.py -> logger related codes
		- exception
			- __init__.py -> expception related codes




### Workflows

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the dvc.yaml



## Finally run the following command

```bash
python app.py
or
streamlit run app_streamlit.py
```
```bash
open up you local host and port
```


## Deployment
### Docker
Build an Image from a Dockerfile
```bash
docker build -t aml-streamlit-app .
```
List local images
```bash
docker images
```
Delete an image
```bash
docker rmi <image_name>
```
Remove all unused images
```bash
docker image prune
```
Run a container with and publish a container’s port(s) to the host.
```bash
docker run -p 8501:8501 aml-streamlit-app
or
docker run -p 5000:5000 app_name
```
```bash
open up your local host and port
```


### AWS-CICD-Deployment-with-Github-Actions 
1. .github\workflows
	- `main.yaml`

2. Login to AWS console

3. Create IAM: `Identity Access Management` user for deployment
	- `AmazonEC2ContainerRegistryFullAccess`
	- `AmazonEC2FullAccess`

4. Create ECR: `Elastic Container registry` to save your docker image in aws
	- ECR Repo URI: 767397970670.dkr.ecr.us-east-1.amazonaws.com/aml_fraud_detector-container

5. Create EC2 (Ubuntu): Virtual machine in the AWS cloud
	- **connect** to EC2 instance
		- Description: About the deployment
		1. Build docker image of the source code
		2. Push  docker image to ECR
		3. Launch EC2 
		4. Pull  image from ECR in EC2

	- Lauch your docker image in EC2
		- Docker setup in EC2
		```bash
		# optional
		sudo apt-get update -y
		sudo apt-get upgrade

		# Required
		curl -fsSL https://get.docker.com -o get-docker.sh
		sudo sh get-docker.sh
		sudo usermod -aG docker ubuntu
		newgrp docker
		```

6. Configure EC2 as self-hosted runner in GitHub
	- Now, Go to GitHub 
```bash
	setting > actions > runner > new self hosted runner > choose os (Linux) > then run command one by one
```
	
7. Setup github secrets:
```bash
	setting > Secrets and variables > actions > New repository secret (in main screen)	

		AWS_ACCESS_KEY_ID=
		AWS_SECRET_ACCESS_KEY=
		AWS_REGION = us-east-1
		AWS_ECR_LOGIN_URI = 767397970670.dkr.ecr.us-east-1.amazonaws.com
		ECR_REPOSITORY_NAME = aml_fraud_detector-container			
```



	

	













### MLflow

- [Documentation](https://mlflow.org/docs/latest/index.html)

- [MLflow tutorial](https://youtube.com/playlist?list=PLkz_y24mlSJZrqiZ4_cLUiP0CBN5wFmTb&si=zEp_C8zLHt1DzWKK)

##### cmd
- mlflow ui

### DagsHub
[DagsHub](https://dagshub.com/)

MLFLOW_TRACKING_URI=https://dagshub.com/robinyUArizona/DL-EndToEnd-Project.mlflow \
MLFLOW_TRACKING_USERNAME=robinyUArizona \
MLFLOW_TRACKING_PASSWORD=292f6d1bdcfe7ac283b512eb8f2fccfce1733a51 \
python script.py

Run this to export as env variables:

```bash

export MLFLOW_TRACKING_URI=https://dagshub.com/robinyUArizona/DL-EndToEnd-Project.mlflow

export MLFLOW_TRACKING_USERNAME=robinyUArizona 

export MLFLOW_TRACKING_PASSWORD=292f6d1bdcfe7ac283b512eb8f2fccfce1733a51

```



### DVC 
[DVC](https://dvc.org/)
Open-source version control system for Data Science and Machine Learning projects. Git-like experience to organize your data, models, and experiments.
1. dvc init
2. dvc repro
3. dvc dag




```bash
Author: Robins Yadav
Data Scientist

```