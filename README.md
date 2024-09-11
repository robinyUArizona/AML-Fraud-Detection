# Anti Money Laundering (AML) Fraud Detection
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
	create a new repository on GitHub or create a new directory, and initialize it
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
	```



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



### Codebase Workflow
```AML Fraud Detection
	- components
		- data_ingestion.py
		- data_transformation.py
		- model_trainer.py
		- model_evaluation.py
		- model_pusher.py
	- configuration
		- s3_operations.py
	- constant
	- entity
		- artifact_entity.py
		- config_entity.py
	- exception
	- logger
	- pipeline
		- train_pipeline.py
		- prediction_pipeline.py
	- utils 
		- main_utils.py
	- ml
		- model.py
```


## How to run?
### STEPS:

Clone the repository

```bash
https://github.com/entbappy/End-to-end-Text-Summarization
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n summary python=3.8 -y
```

```bash
conda activate summary
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up you local host and port
```


```bash
Author: Robins Yadav
Data Scientist

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


### About MLflow & DVC

MLflow

 - Its Production Grade
 - Trace all of your expriements
 - Logging & taging your model


DVC 

 - Its very lite weight for POC only
 - lite weight expriements tracker
 - It can perform Orchestration (Creating Pipelines)



## AWS-CICD-Deployment-with-Github-Actions

### 1. Login to AWS console.

### 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
### 3. Create ECR repo to store/save docker image
    - Save the URI: 566373416292.dkr.ecr.us-east-1.amazonaws.com/chicken

	
### 4. Create EC2 machine (Ubuntu) 

### 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
### 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


## 7. Setup github secrets:

    AWS_ACCESS_KEY_ID=

    AWS_SECRET_ACCESS_KEY=

    AWS_REGION = us-east-1

    AWS_ECR_LOGIN_URI = demo>>  566373416292.dkr.ecr.ap-south-1.amazonaws.com

    ECR_REPOSITORY_NAME = simple-app
```


```bash
pip install -r requirements.txt
```

```bash
python app.py
```

```bash
Now open up your local host 0.0.0.0:8080
```