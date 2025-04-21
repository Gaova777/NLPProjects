from transformers import pipeline
from nltk.tokenize import sent_tokenize
import torch
import numpy as np
import pandas as pd
import nltk
import os
import sys
import pathlib
folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path, "../"))
from utils import load_subtitles_dataset

nltk.download('punkt')
nltk.download('punkt_tab')

class ThemeClassifier():
    def __init__(self, theme_list):
        self.model_name = "facebook/bart-large-mnli"
        self.device = 0 if torch.cuda.is_available() else -1
        self.theme_list = theme_list
        self.theme_classifier = self.load_model(self.device)

    def load_model(self, device):
        return pipeline(  
            "zero-shot-classification",
            model=self.model_name,
            device=device
        )
    
    def get_theme_inference(self,script):
        script_sentences = sent_tokenize(script)

        # Batch Sentence
        sentence_batch_size = 20
        script_batches = []
        for index in range(0, len(script_sentences), sentence_batch_size):
            sent = " ".join(script_sentences[index:index + sentence_batch_size])
            script_batches.append(sent)
        
        # Run Model
        theme_output = self.theme_classifier(
            script_batches[:2],
            self.theme_list,
            multi_label=True
        )

        # Wrangle output
        theme = {}
        for output in theme_output:
            for label, score in zip(output['labels'], output['scores']):
                if label not in theme:
                    theme[label] = []
                theme[label].append(score)
        theme = {key: np.mean(np.array(value)) for key, value in theme.items()}

        return theme

    def get_theme(self,dataset_path,save_path=None):
        #Read Save Output if Exist
        if save_path is not None and os.path.exists(save_path):
            return pd.read_csv(save_path)
        #loadModel
        df = load_subtitles_dataset(dataset_path)
        df = df.head(2)
        #RUn Inference
        output_themes = df['script'].apply(self.get_theme_inference)

        themes_df = pd.DataFrame(output_themes.tolist())
        df[themes_df.columns] = themes_df

        #save output
        if save_path is not None:
            df.to_csv(save_path,index=False)

        return df