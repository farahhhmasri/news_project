# 📢 News Project  

🚀 A comprehensive news classification project with training experiments and containarized application.  

<br>
<br>

## 📌 Project Features  

✅ This project detects the sentiment of the provided news statements. Positive news will have a socre close to +1 or higher, negative news will have a score close to -1 or lower and neutral news will be in-between. <br>
For this task the used model is **twitter-roberta-base-sentiment**. For in-detail information about it, please check [hugging face roberta](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment). In addition to classifying based on the news context, date of the news is also taken into consideration when generating scores, as newer news will have more effect than older news.<br>
The output is provided as .json file in which the news are sorted descendingly based on their scores.<br><br>

✅ There are multiple training expirements for different models trained on the dataset to get the best performance model. Training experiments are tracked using mlflow for better management and versioning.<br><br>

✅ Two options of deployment; either using FastAPI or by building a Docker image.<br>
Out-of-box docker image is also available for quicker deployment.  

<br>

## 🛠️ Installation and usage
<small>*Please note that these commands are tested with Windows, in Linux it may differ. Change accordingly.*</small>

### ✔ Option one 
  1. Clone the repository:  
  ```git clone https://github.com/farahhhmasri/news_project.git```

  2. install requirement:  
  ```pip install -r requirements.txt```

  3. Move to source code:
  ```cd src```

  3. Run main.py to run FastAPI:
  ```python main.py```

  4. **Check the provided request example in the [test](https://github.com/farahhhmasri/news_project/tree/3de56dfca923fca912b1a7f4f9bc44989fdc69c1/test/request_example) directory to get insights on how your request should be.**

  5. Once you send a request, three folders will be created logs, output and sys_stats.

  - **logs** contain the generated system logs which are essential for debugging.
  - **output** contains the output of the model's classification.
  - **sys_stats** contains the system metrics such as, when the request was sent, response time, the used and available memory. 

### ✔ Option Two
Pull a pre-created docker image that has the application containerized.
  1. Run ```docker pull farahmasri/news-deploy:latest```

  2. Run the image and add its needed volumes by running: 
  ```docker run -d -p 8000:8000 -v "$pwd/logs:/app/logs" -v "$pwd/output:/app/output" -v "$pwd/sys_stats:/app/sys_stats" farahmasri/news-deploy:latest```

  - **-d** runs the container in detached mode.
  - **-p 8000:8000** makes the container's port 8000 accessible on the host machine's port 8000.
  - **-v "$pwd/logs:/app/logs"** creates a volume mount from host current directory and container /app/logs
  - **-v "$pwd/output:/app/output"** creates a volume mount from host current directory and container /app/output
  - **-v "$pwd/output:/app/sys_stats"** creates a volume mount from host current directory and container /app/sys_stats



### ✔ Checking training expirements: <br>
You can run the jupyter notebooks that are provided in [notebooks](https://github.com/farahhhmasri/news_project/tree/3de56dfca923fca912b1a7f4f9bc44989fdc69c1/notebooks/training_experiments) folder to get insights about the training experiments.
