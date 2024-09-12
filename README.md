# Anti Money Laundering (AML) Fraud Detection

## Project Overview
The AML Fraud Detection project aims to build a machine learning system to identify potentially fraudulent transactions. Money laundering is a multi-billion dollar issue. This project is crucial for financial institutions to enhance their Anti-Money Laundering (AML) systems and reduce the occurrence of both false positives (legitimate transactions flagged as fraudulent) and false negatives (fraudulent transactions that go undetected).

### Solution Approach
Machine Learning: ML Classification Algorithms

## Project Lifecycle
		1. Understanding the Problem Statement
		2. Data Collection
		3. Data Checks to perform
		4. Exploratory Data Analysis (EDA) 
		5. Data Pre-Processing and Feature Engineering
		6. Model Evaluation and Training - Fit the ML classification algorithm and find out which one performs better
		7. Choose Best Model based on desired metrics
		8. Model deployment with CI/CD pipeline
		9. Web interactive interface
		10. Outcomes

### Understanding the Problem Statement
The problem involves detecting money laundering activities, which are challenging because most automated algorithms have a high false positive rate: legitimate transactions incorrectly flagged as laundering. The converse is also a major problem -- false negatives, i.e. undetected laundering transactions. 

### Data Collection
Data was sourced from Kaggle, providing a comprehensive dataset of transactions labeled as fraudulent (1) or legitimate (0).

Data Source: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/data

### Data Checks to Perform
Initial data validation included checking for missing values, inconsistencies, and anomalies to ensure data integrity.

### Exploratory Data Analysis (EDA)
In-depth EDA was conducted to understand the distribution of features, detect patterns, and identify correlations between variables. Visualization techniques were employed to uncover hidden insights.

### Data Pre-Processing and Feature Engineering
Data was cleaned and transformed to prepare it for model training. This included:
- **Handling Missing Values:** Addressed any gaps in the data to ensure completeness.
- **Encoding Categorical Variables:** Converted categorical variables into numerical formats, as machine learning models require numerical input.
- **Normalizing Numerical Features:** Scaled numeric variables to ensure uniformity and improve model performance.

Feature engineering was employed to create new variables that could enhance model performance. This process involved:
- **Scaling Numeric Variables:** Ensured that numeric features were on a similar scale to prevent any one feature from dominating the model.
- **Encoding Categorical Variables:** Transformed categorical data into numerical representations, enabling the model to process these features effectively.

#### Multicollinearity Check
- **Numerical Features:** Variance Inflation Factor (VIF) was calculated to identify multicollinearity among numerical features. Features with high VIF values were considered for removal or transformation.
- **Categorical Features:** Chi-squared tests were conducted to evaluate the relationship between categorical features and the target variable. The null hypothesis (no association between features) and the alternative hypothesis (association exists) were tested to assess feature importance and avoid redundancy.


### Model Building and Evaluation
Multiple machine learning models were built using classification algorithms, including Random Forest, AdaBoost, and XGBoost. Hyperparameter tuning was performed using grid search and cross-validation to optimize model performance.

### Model Selection
The best-performing models were selected based on accuracy, precision, recall, and F1-score metrics. The selected model was further tuned and validated on the test dataset.

### Deployment
The final model was containerized using Docker and deployed on AWS. The deployment process involved the following steps:

**Notes**: 
Finally run the following command to test in local after buling prediction pipeline for web interface using FastAPI or streamlit
```bash
python app.py
or
streamlit run app_streamlit.py
```
```bash
open up you local host and port
```


		
**Docker** Notes

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


1. **CI/CD Setup with GitHub Actions:**
   - Configured GitHub Actions workflows in the `.github/workflows` directory, the `main.yaml` file, to automate the build, test, and deployment stages.

2. **AWS Console Setup:**
   - **Login to AWS Console:** Accessed the AWS Management Console to set up required resources.
   - **IAM User Creation:** Created an IAM user with permissions for deployment:
     - `AmazonEC2ContainerRegistryFullAccess`: Full access to Amazon Elastic Container Registry (ECR).
     - `AmazonEC2FullAccess`: Full access to Amazon EC2.

3. **Elastic Container Registry (ECR) Setup:**
   - Created an ECR repository for storing Docker images.
   - **ECR Repo URI:** `767397970670.dkr.ecr.us-east-1.amazonaws.com/aml_fraud_detector-container`

4. **EC2 Instance Setup:**
   - Create EC2 (Ubuntu): Virtual machine in the AWS cloud
   - Launched an EC2 instance (Ubuntu) to run the application.
   - **Connect to EC2 Instance:** Built Docker image, pushed it to ECR, and launched it on EC2.
		- **Description**: About the deployment
		1. Build docker image of the source code
		2. Push  docker image to ECR
		3. Launch EC2 
		4. Pull  image from ECR in EC2

	- Lauch docker image in EC2
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

5. Configure EC2 as self-hosted runner in GitHub
	- Now, Go to GitHub 
```bash
	setting > actions > runner > new self hosted runner > choose os (Linux) > then run command one by one
```
	
6. Setup github secrets:
```bash
	setting > Secrets and variables > actions > New repository secret (in main screen)	

		AWS_ACCESS_KEY_ID=
		AWS_SECRET_ACCESS_KEY=
		AWS_REGION = us-east-1
		AWS_ECR_LOGIN_URI = 767397970670.dkr.ecr.us-east-1.amazonaws.com
		ECR_REPOSITORY_NAME = aml_fraud_detector-container			
```



## Web Interfaces
Two web interfaces were developed to interact with the model:
- **FastAPI:** A high-performance web framework for building APIs with Python, used to create an API endpoint for real-time fraud detection.
- **Streamlit:** An open-source app framework for machine learning and data science projects, used to build an interactive web application for visualizing results and interacting with the model.

## Tools and Technologies
- **Data Processing:** Python, Pandas, NumPy, Scikit-learn
- **Modeling:** Random Forest, XGBoost, Logistic Regression
- **Hyperparameter Tuning:** Grid Search, Cross-Validation
- **Deployment:** Docker, AWS (EC2, ECR), GitHub Actions, MLflow, DVC
- **Web Interfaces:** FastAPI, Streamlit
- **Version Control and Experiment Tracking:** GitHub, DVC, MLflow

## Outcome

The project resulted in a robust machine learning model that significantly enhanced the detection of fraudulent transactions, effectively reducing both false positives and false negatives. The model was successfully deployed using Docker on AWS, leveraging ECR and EC2 instances, and was integrated into a CI/CD pipeline with GitHub Actions to ensure continuous updates and monitoring.

MLflow was used for experiment tracking, allowing the team to log and compare different model versions and their performance metrics. It provided valuable insights into model evaluation by tracking parameters, metrics, and artifacts, which aided in making informed decisions during model tuning and selection.

The web interfaces enabled users to easily determine whether a transaction is fraudulent or legitimate based on its details. These interfaces were developed using FastAPI for API-based access and Streamlit for an interactive, web-based experience, ensuring they were user-friendly and flexible.



```bash
Author: Robins Yadav
Data Scientist

```



#### Project setup Notes
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



#### Workflows 

In progress
1. constants
2. entity


1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the dvc.yaml



#### MLflow
- [Documentation](https://mlflow.org/docs/latest/index.html)
```bash
- mlflow ui
```

#### DagsHub
[DagsHub](https://dagshub.com/) 

MLflow is used for tracking experiments, logging metrics, parameters, and artifacts. Integrating DagsHub with MLflow allows you to track machine learning experiments and their associated data and models in one place.

MLFLOW_TRACKING_URI=https://dagshub.com/robinyUArizona/DL-EndToEnd-Project.mlflow \
MLFLOW_TRACKING_USERNAME=robinyUArizona \
MLFLOW_TRACKING_PASSWORD=292f6d1bdcfe7ac283b512eb8f2fccfce1733a51 \
python script.py

Run this to export as env variables:

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/robinyUArizona/DL-EndToEnd-Project.mlflow

export MLFLOW_TRACKING_USERNAME=robinyUArizona 

export MLFLOW_TRACKING_PASSWORD=
```

#### DVC 
[DVC](https://dvc.org/)
Open-source version control system for Data Science and Machine Learning projects. Git-like experience to organize your data, models, and experiments.
```bash
1. dvc init
2. dvc repro
3. dvc dag
```


```bash
Author: Robins Yadav
Data Scientist

```