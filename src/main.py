from model_utils.predictions_generator import GeneratePredictions
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
from starlette.requests import Request
import datetime
import pandas as pd
import os
import json
import logging
import psutil
import time



current_date = datetime.date.today().isoformat()  
log_dir = f'../logs/{current_date}'
os.makedirs(log_dir, exist_ok=True)
current_time = datetime.datetime.now().strftime("%H-%M-%S")
log_file = f"{log_dir}/{current_time}.log"

logger = logging.getLogger(__name__)
logging.basicConfig(filename=log_file, encoding='utf-8', level=logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.WARNING)


sys_log_dir = f'../sys_stats/{current_date}'
os.makedirs(sys_log_dir, exist_ok=True)
sys_log_file = f"{sys_log_dir}/{current_time}.json"

def get_system_stats(start_time, inference_time):
    sys_metrics =  {
        "request_time": datetime.datetime.fromtimestamp(start_time).isoformat(),
        "time_to_repond" : f"{inference_time:.4f} seconds",
        "cpu_percent": psutil.cpu_percent(),
        "memory_used": f"{psutil.virtual_memory().used / (1024 ** 3):.2f} GB",  
        "memory_available": f"{psutil.virtual_memory().available / (1024 ** 3):.2f} GB",
    }
    with open(sys_log_file, "a") as f:
        f.write(json.dumps(sys_metrics) + "\n")  # Append new metrics as a new line
    
    return f"System metrics logged successfully in {sys_log_file}!"

model = GeneratePredictions()

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    logging.info("API is running and ready to accept requests!")
    return templates.TemplateResponse("index.html", {"request": request, "title": "FastAPI Homepage"})

@app.post("/predict")
async def predict(request: Request):
    json_data = await request.json()
    logging.info("A predict request was sent to the API.")
    start_time = time.time()
    try: 
        df = pd.DataFrame(json_data)
        if 'id' not in df.columns:
            logging.error("Missing 'id' column. Predicitions will not be generated.")
            raise HTTPException(status_code=400, detail="Missing 'news' column.")
        if 'date' not in df.columns:
            logging.error("Missing 'date' column. Predicitions will not be generated.")
            raise HTTPException(status_code=400, detail="Missing 'news' column.")
        if 'news' not in df.columns:
            logging.error("Missing 'news' column. Predicitions will not be generated.")  
            raise HTTPException(status_code=400, detail="Missing 'news' column.")
        predictions = model.predict(df)
        inference_time = time.time() - start_time
        sys_stats = get_system_stats(start_time, inference_time)
        logging.info(sys_stats)
        outputpath = f"../output/{current_date}"
        os.makedirs(outputpath, exist_ok=True)
        predictions.to_json(f'{outputpath}/{current_time}.json', orient='records')
        logging.info(f"Output is saved to {outputpath}/{current_time}.")
        return {"predictions": predictions.values.tolist()}

    except Exception as e:
        logging.error(f"An error occured inside the API predict function: {e}.")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, port=8000, host="0.0.0.0")
    