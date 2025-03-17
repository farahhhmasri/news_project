from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import logging
import re
import os


class GeneratePredictions:

    def __init__(self):
        # logging.info("Inside predicitions_generator.py - Loading Roberta Model...")
        model_dir = Path(__file__).resolve().parent / "pretrained_roberta"
        tokenizer_path = model_dir / "tokenizer"
        model_path = model_dir / "model"
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_path), local_files_only=True)

    def clean_text(self, text):
        try: 
            if text[0] == 'b':
                text= text[1:]
            if text[0] == '"' or text[0] == "'":
                text = text[1:len(text)-1]  
            text = re.sub(r'\\+', r'\\', text)
            text = re.sub(r'(?<!\d)\\(?!\d)', '', text)
            return text
        except Exception as e:
            logging.warning(f"An error occurred inside clean_text function when handling this text: {text}")
            logging.warning(f"Error  inside clean_text function: {e} ")
    
    def polarity_scores_roberta(self, text, date, year):
        try: 
            encoded_text = self.tokenizer(text, return_tensors='pt')
            output = self.model(**encoded_text)
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)
            sentiment_score = scores[2] - scores[0] + 0.5 * scores[1]
            scores_dict = {
                'neg' : scores[0],
                'neu' : scores[1],
                'pos' : scores[2],
                'sentiment_score' : sentiment_score,
                'date': date,
                'score': (sentiment_score) * 5 + (1 / (2025 - year + 1)) * 10
            }
            return scores_dict
        except Exception as e:
            logging.warning(f"An error occurred inside polarity_scores_roberta function when handling this text: {text}")
            logging.error("An error occurred inside polarity_scores_roberta function: {e}")
    
    def generate_polarity_scores(self, df):
        logging.info("Running generate_polarity_scores function...")
        res = {}
        for i, row in tqdm(df.iterrows(), total=len(df)):
            try:
                text = self.clean_text(row['news'])
                myid = row['id']
                date = row['date']
                year = pd.to_datetime(row['date']).year
                result = self.polarity_scores_roberta(text, date, year)
                res[myid] = result
            except Exception as e:
                logging.warning(f"An error occurred inside generate_polarity_scores function when handling this row {myid}, {text}")
                logging.error(f"An error occurred inside generate_polarity_scores: {e}")
        logging.info("Generated polarity scores using roberta model successfully!")
        return res
    
    def predict(self, df):
        logging.info("Running predict function...")
        try: 
            result = self.generate_polarity_scores(df)
            logging.info("Predictions are created.")
            sorted_news = dict(sorted(result.items(), key=lambda item: item[1]['score'], reverse=True))
            scores = []
            predicted_labels = []
            for news_id in sorted_news:
                news_score = result.get(news_id).get('score', 0)
                scores.append(news_score)
                label = lambda x : 'POSITIVE' if x > 0 else 'NEGATIVE'
                predicted_labels.append(label(news_score))
            logging.info("Sorted the news based on the generated predictions.")
            sorted_ids = list(sorted_news.keys())
            sorted_df = df.set_index('id').loc[sorted_ids]
            sorted_df['score'] = scores
            sorted_df['prediction'] = predicted_labels
            return sorted_df
        except Exception as e:
            logging.error(f"An Error occured when calling predict function: {e}.")
        

