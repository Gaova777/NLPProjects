import torch
import huggingface_hub
from transformers import (AutoTokenizer, 
                          AutoModelForSequenceClassification,
                          DataCollatorWithPadding,
                          TrainingArguments,
                          pipeline)
import pandas as pd
from .cleaner import Cleaner
from sklearn import preprocessing
from sklearn.model_selection import train_test_split #That is used to split the dataset into training and testing sets.
from datasets import Dataset
from .training_utils import get_class_weights, compute_metrics
from .custom_trainer import CustomTrainer
import gc

'''
     This is a class that is used to classify jutsu using a transformer model.

'''

class JutsuClassifier():
    def __init__(self,
                 model_path,
                 data_path=None,
                 text_column_name='text',
                 label_column_name='jutsu',
                 model_name = "distilbert/distilbert-base-uncased",
                 test_size=0.2,
                 num_labels=3,
                 huggingface_token=None):
        
        self.model_path = model_path
        self.data_path = data_path
        self.text_column_name = text_column_name
        self.label_column_name = label_column_name
        self.test_size = test_size
        self.model_name = model_name
        self.num_labels = num_labels
        self.huggingface_token = huggingface_token
        self.device = 'cuda' if torch.cuda.is_available() else '-1'

        '''
        After to create the properties and the constructor of the class, we check if the model path exists in the huggingface hub. In the next line codes we check if the huggingface token is provided, if it is provided we login to the huggingface hub.
        If the model path does not exist in the huggingface hub, we check if the data path is provided. If it is not provided, we raise a value error. If the data path is provided, we load the data and train the model.
        '''

        if self.huggingface_token is not None:
            huggingface_hub.login(self.huggingface_token)

        self.tokenizer = self.load_tokenizer()
        if not huggingface_hub.repo_exists(self.model_path):

            #Validate the data and return an error if the data path is not provided
            if data_path is None:
                raise ValueError("Data path is required to train the model, the model path doesnt exist in huggingface hub")
            
            '''
            after validating the data path, we load the data with the load_data method, then we convert the data to pandas dataframe and concatenate the train and test data.
            After that we reset the index because the index is not important in this case and we get the class weights using the get_class_weights method.(See the file if you want to know more about this method)

            After that we train the model with the train_model method, passing the train data, test data and class weights as parameters.
            '''
            train_data, test_data = self.load_data(self.data_path)
            train_data_df = train_data.to_pandas()
            test_data_df = test_data.to_pandas()
            
            all_data = pd.concat([train_data_df,test_data_df]).reset_index(drop=True)
            class_weights = get_class_weights(all_data)

            self.train_model(train_data, test_data, class_weights)


        '''
        In this Section we just apply the load_model. We This will load the model and tokenizer from the specified path and return a pipeline object

        '''


        self.model = self.load_model(self.model_path)
    
    def load_model(self, model_path):
        model = pipeline('text-classification', model=model_path, return_all_scores=True) 
        return model
    
    '''
        🟠 The train_model method is part of the trainning pipeline of a classify model ussing huggingface transformers. It takes the train and test data, class weights as input and trains the model using the CustomTrainer class. Inside of the method we decribe all most important parts.

    '''
    def train_model(self,train_data,test_data,class_weights):
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels = self.num_labels, id2label=self.label_dict)

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)    
        training_args = TrainingArguments(
            output_dir=self.model_path,
            learning_rate=2e-4,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            push_to_hub=True,
        )


        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=test_data,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics = compute_metrics,
        )

        trainer.set_device(self.device)
        trainer.set_class_weights(class_weights)

        trainer.train()

        #Flush memory

        del trainer, model
        gc.collect()

        if self.device == 'cuda':
            torch.cuda.empty_cache()
    def simplify_jutsu(self, jutsu):
        if "Genjutsu" in jutsu:
            return "Genjutsu"
        if "Taijutsu" in jutsu:
            return "Taijutsu"
        if "Ninjutsu" in jutsu:
            return "Ninjutsu"
    
    def preprocess_function(tokenizer,examples):
        return tokenizer(examples['text_cleaned'],truncation=True)
 
    def load_data(self, data_path):
        df = pd.read_json(data_path,lines=True)
        df["jutsu_type_simplified"] = df['jutsu_type'].apply(self.simplify_jutsu)
        df['text'] = df['jutsu_name']+ ". " + df['jutsu_description']
        df[self.label_column_name] = df['jutsu_type_simplified']
        df = df[['text', self.label_column_name]]
        df = df.dropna()
        
        #clean Data
        cleaner = Cleaner()
        df['text_cleaned'] = df[self.text_column_name].apply(cleaner.clean)
        #Encode labels
        le = preprocessing.LabelEncoder()
        le.fit(df[self.label_column_name].to_list())
        label_dict = {index:label_name for index,label_name in enumerate(le.__dict__['classes_'].tolist())}
        self.label_dict = label_dict
        df['label'] = le.transform(df[self.label_column_name].to_list())

        #Train / Test 

        test_size = 0.2
        df_train, df_test = train_test_split(df,
                                             test_size=test_size,
                                             stratify=df['label'])
                                            
            #Convert Pandas To a Hugging face dataset
        train_dataset = Dataset.from_pandas(df_train)
        test_dataset = Dataset.from_pandas(df_test)

        #Tokenize Dataset
        tokenized_train = train_dataset.map(lambda examples: self.preprocess_function(self.tokenizer,examples),
                                            batched=True)

        tokenized_test = test_dataset.map(lambda examples: self.preprocess_function(self.tokenizer,examples),
                                            batched=True)

        return tokenized_train, tokenized_test


    def load_tokenizer(self):
        if huggingface_hub.repo_exists(self.model_path):
             tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        else:
             tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        return tokenizer
                
    def postprocess(self, model_output):
        output = []
        for pred in model_output:
            label = max(pred, key=lambda x: x['score'])['label']
            output.append(label)
        return output
    
    def classify_jutsu(self,text):
        model_output = self.model(text)
        preficitions = self.postprocess(model_output)
        return preficitions