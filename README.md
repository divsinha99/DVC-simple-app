### Steps for DVC and MLOPS

* Create the virtual environment. In cmd line, run the below command : 
```bash
conda create -n wineq python=3.7 -y
```

* Activating the environment : 
```
conda activate wineq
```
    
* Create a Requirements.txt file  :
```
touch Requirements.txt
```
    
* Add the following libraries in Requirements.txt file : 
```		
dvc
dvc[gdrive]
sklearn
```
    
* Go to command prompt and run the below command :
```
pip install -r Requirements.txt
```    
* Store the streaming/raw data in data_given folder.
    
* Create template file : 
```
touch template.py
```    
* Inside template.py, write the following lines of code :
```
import os
dirs = [
	os.path.join("data", "raw"),
	os.path.join("data","processed"),
	"notebooks",
	"saved_models",
	"src",
	"report"
	]

for dir_ in dirs:
	os.makedirs(dir_, exist_ok=True)
	with open(os.path.join(dir_, ".gitkeep"),"w") as f:
	    pass


files = [
	"dvc.yaml",
	"params.yaml",
	".gitignore",
	"README.md",
	os.path.join("src", "__init__.py")
	]

for file_ in files:
	with open(file_,"w") as f:
    		pass
```

## Create Params.yaml file (config file for our pipeline)

    base:
      project: winequality-project
      random_state: 42
      target_col: TARGET

    data_source:
      s3_source: data_given/winequality.csv

    load_data:
      raw_dataset_csv: data/raw/winequality.csv

    split_data:
      train_path: data/processed/train_winequality.csv
      test_path: data/processed/test_winequality.csv
      test_size: 0.2

    estimators:
      ElasticNet:
        params:
          # alpha: 0.88
          # l1_ratio: 0.89
          alpha: 0.9
          l1_ratio: 0.4
    model_dir: saved_models

## Create get_data.py under src/ folder
For fetching raw files from S3 bucket or streaming data.

    ## Read paramaters
    ## Process
    ## Return Dataframe

    import os
    import yaml
    import pandas as pd
    import argparse

    def read_params(config_path):
        with open(config_path) as yaml_file:
            config = yaml.safe_load(yaml_file)
        return config

    def get_data(config_path):
        config = read_params(config_path)
        #print(config)
        data_path = config["data_source"]["s3_source"]
        df = pd.read_csv(data_path, sep=",", encoding='utf-8')
        return df



    if __name__=="__main__":
        args = argparse.ArgumentParser()
        args.add_argument("--config", default="params.yaml")
        parsed_args = args.parse_args()
        data = get_data(config_path=parsed_args.config)

## Create load_data.py under src/
For loading data from S3 bucket to raw folder in data folder.

    ## Read paramaters
    ## Process
    ## Return Dataframe

    import os
    import yaml
    import pandas as pd
    import argparse

    def read_params(config_path):
        with open(config_path) as yaml_file:
            config = yaml.safe_load(yaml_file)
        return config

    def get_data(config_path):
        config = read_params(config_path)
        #print(config)
        data_path = config["data_source"]["s3_source"]
        df = pd.read_csv(data_path, sep=",", encoding='utf-8')
        return df



    if __name__=="__main__":
        args = argparse.ArgumentParser()
        args.add_argument("--config", default="params.yaml")
        parsed_args = args.parse_args()
        data = get_data(config_path=parsed_args.config)

## Create dvc.yaml
For the different stages of our pipeline.

	stages:
      load_data:
        cmd: python src/load_data.py --config=params.yaml
        deps:
        - src/get_data.py
        - src/load_data.py
        - data_given/winequality.csv
        outs:
        - data/raw/winequality.csv

## Run dvc repro command
To lock the dvc files for all the stages. dvc lock files will sense any kind of change in the repository and update the lock file accordingly.

	dvc repro
	
## Update in github

	git add . && git commit -m "Stage 1 Complete"
	git push origin main
	
## Create Split_data.py under src folder
To split the raw data into train and test data and store in data/processed folder.

	## Splitting the data into train and test set
	## Save it in the processed folder

	import os
	import argparse
	import pandas as pd
	from sklearn.model_selection import train_test_split
	from get_data import read_params

	def split_and_save(config_path):
	    config = read_params(config_path)
	    train_data_path = config["split_data"]["train_path"]
	    test_data_path = config["split_data"]["test_path"]
	    raw_data_path = config["load_data"]["raw_dataset_csv"]
	    split_ratio = config["split_data"]["test_size"]
	    random_state = config["base"]["random_state"]

	    df = pd.read_csv(raw_data_path, sep=",")
	    train, test = train_test_split(
		df, 
		test_size=split_ratio, 
		random_state=random_state)

	    train.to_csv(train_data_path, sep=",", index=False ,encoding="utf-8")
	    test.to_csv(test_data_path, sep=",", index=False ,encoding="utf-8")


	if __name__=="__main__":
	    args = argparse.ArgumentParser()
	    args.add_argument("--config", default="params.yaml")
	    parsed_args = args.parse_args()
	    split_and_save(config_path=parsed_args.config)    
    
## Append the stage 2 in dvc.yaml file 
Stage 2 is splitting the raw data into train and test data and store into data/processed folder. Update dvc file accordingly.
	
	  split_data:
	    cmd: python src/split_data.py --config=params.yaml
	    deps:
	    - src/split_data.py
	    - data/raw/winequality.csv
	    outs:
	    - data/processed/train_winequality.csv
	    - data/processed/test_winequality.csv  

## Run dvc repro command to run the 2nd stage in pipeline

	dvc repro
	
## Push the changes into Github 
	
	git add . && git commit -m "Stage 2 Complete"
	git push origin main

## Create train_and_evaluate.py in src folder
To train the algorithm and store the scores and model params.

	## load train and test set
	## train algorithm
	## save the metrices and params

	import os
	import warnings
	import sys
	import pandas as pd
	import numpy as np
	from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
	from sklearn.model_selection import train_test_split
	from sklearn.linear_model import ElasticNet
	from urllib.parse import urlparse
	from get_data import read_params
	import argparse, joblib, json

	def eval_metrics(actual, pred):
	    rmse = np.sqrt(mean_squared_error(actual, pred))
	    mae = mean_absolute_error(actual, pred)
	    r2 = r2_score(actual, pred)
	    return rmse, mae, r2


	def train_and_evaluate(config_path):
	    config = read_params(config_path)
	    train_data_path = config["split_data"]["train_path"]
	    test_data_path = config["split_data"]["test_path"]
	    random_state = config["base"]["random_state"]
	    model_dir = config["model_dir"]

	    alpha = config["estimators"]["ElasticNet"]["params"]["alpha"]
	    l1_ratio = config["estimators"]["ElasticNet"]["params"]["l1_ratio"]    

	    target = config["base"]["target_col"]

	    train = pd.read_csv(train_data_path, sep=",")
	    test = pd.read_csv(test_data_path, sep=",")

	    train_y = train[target]
	    test_y = test[target]

	    train_x = train.drop(target, axis=1)
	    test_x = test.drop(target, axis=1)

	    lr = ElasticNet(alpha=alpha, 
			    l1_ratio=l1_ratio, 
			    random_state=random_state)

	    lr.fit(train_x, train_y)

	    predicted_qualities = lr.predict(test_x)

	    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

	    print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
	    print("  RMSE: %s" % rmse)
	    print("  MAE: %s" % mae)
	    print("  R2: %s" % r2)    

	########################################################################

	    scores_file = config["reports"]["scores"]
	    params_file = config["reports"]["params"]

	    with open(scores_file, "w") as f:
		scores = {
		    "rmse": rmse,
		    "mae": mae,
		    "r2": r2
		}
		json.dump(scores, f, indent=4)

	    with open(params_file, "w") as f:
		params = {
		    "alpha": alpha,
		    "l1_ratio": l1_ratio,
		}
		json.dump(params, f, indent=4)
	#####################################################


	    os.makedirs(model_dir, exist_ok=True)
	    model_path = os.path.join(model_dir, "model.joblib")

	    joblib.dump(lr, model_path)    


	if __name__=="__main__":
	    args = argparse.ArgumentParser()
	    args.add_argument("--config", default="params.yaml")
	    parsed_args = args.parse_args()
	    train_and_evaluate(config_path=parsed_args.config)    

## Append the below lines in dvc.yaml file


	  train_and_evaluate:
	    cmd: python src/train_and_evaluate.py --config=params.yaml
	    deps:
	    - data/processed/train_winequality.csv
	    - data/processed/test_winequality.csv 
	    - src/train_and_evaluate.py
	    params:
	    - estimators.ElasticNet.params.alpha
	    - estimators.ElasticNet.params.l1_ratio
	    metrics:
	    - report/scores.json:
		cache: false
	    - report/params.json:
		cache: false
	    outs:
	    - saved_models/model.joblib 

## Run dvc repro command to run the 2nd stage in pipeline

	dvc repro
	
## Push the changes into Github 
	
	git add . && git commit -m "Stage 3 Complete" && git push origin main
	
## To show the dvc parameters in cmd line

	dvc metrics show
	
## To compare metrics values from the past

	dvc metrics diff
	
## Adding dependencies pytest, tox
Update Requirements.txt with pytest & tox

## Create a new file tox.ini
Add the below lines in tox.ini :

	[tox]
	envlist = py37, py38
	skipsdist = True

	[testenv]
	deps = -rrequirements.txt
	commands = 
	    pytest -v
	    
Run the command in pytest -v in cmd prompt, it will indicate no test to run as we haven't prepared any test cases.

In order to create test cases, we will create a new folder 
	
	--> tests
		|--> conftest.py
		|--> test_config.py
		|--> __init__.py
		
		
The above files will created under tests folder.

We will create our test cases inside test_config.py and run the command tox which will create a virtual environment and install dependencies and run the test cases :

	tox

If you want to reload the tox environment reinstalling the dependencies, pass the command :

	tox -r
	
Now we will create setup.py file. and run the below command to create a package :

	pip install -e .
	
Build your own package commands :

	python setup.py sdist bdist_wheel

Create directories "prediction_service" and "webapp" as below :

	webapp
	|--> static
		|--> css
			|--> main.css
		|--> script
			|--> index.js
	|--> templates
		|--> 404.html
		|--> base.html
		|--> index.html
	prediction_service
	|--> model
		|--> model.joblib
	|--> __init__.py
	|--> prediction.py
	|--> schema_in.json
	app.py
	
Build the api and a basic predict and api response functions in app.py file. 
Once the api is tested successfully, we need to deploy the app to Heroku.
For deployment we will use github action workflow for deploying the same.

Push to Github and navigate to Actions of your github repository to see the 
successful build status.
```
git add . && git commit -m "Github action workflow without deployment updated" && git push origin main
```

## Deployment to Heroku
### Using Github action workflow for deployment
* let's create a directory ".github/workflows"
```
mkdir -p .github/workflows
```
* Create a file "ci-cd.yaml" under the workflows directory :
```
touch .github/workflows/ci-cd.yaml
```
Whenever we push/pull a request to the main branch, then only this yaml file 
will be executed. It will build a job, create environment, install dependencies,
it will run flake8 for any syntax errors and then it will run pytest. All these actions
will be performed in the github itself. For the moment, comment out deployment
lines of code in ci-cd.yaml file and test it!!!

Note : Changed "on" parameter in ci-cd.yaml file to for the exception - "No 
trigger mentioned in on".
```
on: [push]
```
* For Deployment to Heroku, follow the below steps - 
	* Login to Heroku dashboard.
	* Create new> new application.
	* Give a unique name to your app and create.
	* It will ask for deployment method, Choose Github as deployment method.
	Search for the repository and connect.
	* Select the appropriate branch in github.
	* Under automatic deploys, select the option - Wait for CI to pass before deploy
	(This option will wait for all the jobs to successfully execute before 
	deploying, if any error, it won't deploy)
	* Select "Automatic Deploys" option.

Next, we need to create app secrets in github for this Heroku app. 
Follow the below steps - 
* Copy the Heroku app name.
* Navigate to settings tab of your Github repository.
* Under Options, select Secrets.
* Create new repository secret. Name it as HEROKU_APP_NAME (to be mentioned in ci-cd.yaml file)
Paste the value copied from Heroku app name in prev step and add the secret.
* Next, we have to create an API token. 
	* Go to Heroku account settings.
	* Under Applications tab, create authorization token.
	* Give a description ("Wine quality app"), you can mention time after which 
	you want it to expire (we can leave it blank). Click on create.
	* Copy the token generated.
	* Go to Github Action secrets and create new repository secret.
	* Name it as HEROKU_API_TOKEN (to be mentioned in ci-cd.yaml file)
	* Paste the token key and add the secret.

Once all the settings are done, we will use github action workflow to deploy the 
workflow. Whenever there will be push request, Github will automatically do the testing and 
deploy it to Heroku app. 

We need to create a procfile for that.
```
touch procfile
```
You need to specify which application you need to run over there like below - 
```
web gunicorn app:app
```
Push it to Github once again to see the successful deployment status - 
```
git add . && git commit -m "Deployment to Heroku with github workflow action updated" && git push origin main
```

Navigate to Github action and check the status of deployment. Once deployment is successful,
navigate to Heroku dashboard, open your app, go to settings. 
Under Domains, you will find your app UR. Open that UR and test your app!!!





