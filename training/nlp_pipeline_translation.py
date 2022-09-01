"""
Translation Trick Implementation

The pipeline uses the translation trick to predict the ICD codes of German medical reports.
It takes the translated English reports as input, fine-tunes the pretrained medical models such as BioBERT etc.,
    and makes predictions.
Note: The translation is done in Jupyter Notebook. http://localhost:8888/notebooks/notebooks/Translation.ipynb
"""

import pandas as pd
import numpy as np
from csv import writer
import os
import csv
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit

import torch
import matplotlib.pyplot as plt
from datasets import Dataset, load_metric
import json
from textwrap import wrap

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, \
    TrainingArguments, Trainer, TrainerCallback, TrainerState, \
    TrainerControl, EarlyStoppingCallback, set_seed)
from transformers.optimization import AdamW, get_scheduler, get_polynomial_decay_schedule_with_warmup

from sklearn.metrics import top_k_accuracy_score, classification_report, average_precision_score, \
    confusion_matrix, multilabel_confusion_matrix, roc_auc_score, plot_confusion_matrix

from sklearn.metrics import ConfusionMatrixDisplay

from datetime import date, datetime
from scipy.special import softmax
import os
import ray
import wandb

# Setting the seed so that different runs with same setting produce same results
set_seed(42)

today = date.today()
d1 = today.strftime("%d_%m_%Y")

# import wandb
os.environ["WANDB_SILENT"] = "true"

# Improves debugging when something goes wrong with the GPU
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

wandb.init()

class MyCallback(TrainerCallback):
    "A callback that prints a message at the beginning of training"

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        print('Completed initialization')

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        print('Finished training')

    def on_train_begin(self, args, state, control, **kwargs):
        print("Starting training")

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        print("Starting epoch")

class NLP_pipeline:
    print('----------- Initializing the NLP_pipeline class -----------')
    def __init__(self,
                 model_name,
                 num_training_epochs,
                 lr_scheduler_type,
                 mlm_translation_direct_label,
                 loaded_from_checkpoint,
                 mlm_dataset):
        self.model = None
        self.model_name = model_name
        self.tokenizer = None
        self.trainer = None
        self.lr_scheduler_type = lr_scheduler_type
        self.mlm_dataset = mlm_dataset

        # Data
        self.raw_dataset_train = None
        self.raw_dataset_val = None
        self.raw_dataset_test = None
        self.tokenized_dataset_train = None
        self.tokenized_dataset_val = None
        self.tokenized_dataset_test = None

        # Training parameters
        self.training_args = None
        self.evaluation_metric = None
        self.metric_accuracy = load_metric("accuracy")
        self.metric_f1 = load_metric("f1")
        self.metric_precision = load_metric("precision")
        self.metric_recall = load_metric("recall")

        self.test_metric = load_metric("accuracy")
        self.test_results = None
        self.test_results_as_list = None
        self.num_training_epochs = num_training_epochs
        self.mlm_translation_direct = mlm_translation_direct_label

        # When user chooses to load a model
        self.loaded_tokenizer = None
        self.loaded_model = None

        self.num_labels = 10
        self.output_dir = None
        self.len_train_data = None
        self.loaded_from_checkpoint = loaded_from_checkpoint
        self.tokenizer_name = None
        self.wandb_url = wandb.run.get_url()

        self.icd_code_dict = {'Z80.3': 0,
                              'Z01.6': 1,
                              'N64.4': 2,
                              'N64.5': 3,
                              'D48.6': 4,
                              'Z12.3': 5,
                              'D24': 6,
                              'R92': 7,
                              'Z85.3': 8,
                              'C50.9': 9
                              }

    print('----------- Initialized the NLP_pipeline class -----------')

    # def map_to_icd_codes(self, class_list_as_integer):
    #     class_list_as_string = []
    #     # output = {'class'}
    #
    #     for class_value in class_list_as_integer:
    #         class_as_string = list(self.icd_code_dict.keys())[list(self.icd_code_dict.values()).index(class_value)]
    #         class_list_as_string.append(class_as_string)
    #     # class_list_as_string = [list(self.icd_code_dict.keys())[list(self.icd_code_dict.values()).index(class_value)] for class_value in class_list_as_integer]
    #     return class_list_as_string

    def print_info(self):
        print('........ Model name - {}'.format(self.model_name))
        print('........ Epochs - {}'.format(self.num_training_epochs))

    def load_data(self):
        """
        Assigns to self - train, validation and test data as Transformers Dataset object.
        Updates the num_labels.
        """
        print('----------- Loading the data -----------')
        # Train - validation - test sets
        df_train = pd.read_csv('/home/kaan/mamma-reports_cleaned/translated/train_translated.csv',
                               index_col=0).reset_index(drop=True)
        df_val = pd.read_csv('/home/kaan/mamma-reports_cleaned/translated/val_translated.csv',
                             index_col=0).reset_index(drop=True)
        df_test = pd.read_csv('/home/kaan/mamma-reports_cleaned/translated/test_translated.csv',
                              index_col=0).reset_index(drop=True)

        # Number of classes in the dataset - will be used as a parameter in other methods
        self.num_labels = len(df_train['label'].unique())
        self.len_train_data = df_train.shape[0]

        # Transformers Dataset objects
        self.raw_dataset_train = Dataset.from_pandas(df_train)
        self.raw_dataset_val = Dataset.from_pandas(df_val)
        self.raw_dataset_test = Dataset.from_pandas(df_test)

        # print('Train:{}'.format(df_train.head()))
        # print('Num labels:{}'.format(self.num_labels))
        print('----------- Loaded the data -----------')

    def initialize_model_and_tokenizer(self,
                                       dropout=0.1):
        """
        Initializes the model and tokenizer for the given model name.
        :return:
        """
        print('----------- Initializing the tokenizer and model -----------')

        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name,
                                                                        num_labels=self.num_labels,
                                                                        hidden_dropout_prob=dropout)

        # If we are loading a model from the checkpoint, we need the tokenizer name from config.json file
        if self.loaded_from_checkpoint:
            with open(f"{self.model_name}/config.json", "r") as read_file:
                data = json.load(read_file)
                self.tokenizer_name = data["_name_or_path"]
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)


        print('----------- Initialized the tokenizer and model -----------')

    def tokenize_function(self,
                          examples,
                          padding="max_length",
                          truncation=True):
        """
        Utility function to tokenize the given list of input text data (examples)
        Input data must have a column named "text"
        """
        tokenized_inputs = self.tokenizer(examples["text"],
                                          padding=padding,
                                          truncation=truncation,
                                          max_length=512,
                                          # return_tensors="pt"
                                          )
        return tokenized_inputs


    def tokenize_data(self):
        """
        Takes raw_dataset_train, raw_dataset_val and raw_dataset_test from self.
        Stores the variables in self.
        """
        print('----------- Tokenizing the data -----------')
        self.tokenized_dataset_train = self.raw_dataset_train.map(self.tokenize_function, batched=True)
        self.tokenized_dataset_val = self.raw_dataset_val.map(self.tokenize_function, batched=True)
        self.tokenized_dataset_test = self.raw_dataset_test.map(self.tokenize_function, batched=True)
        # print(self.tokenized_dataset_train[0:2]["input_ids"])
        print('----------- Tokenized the data -----------')

    def investigate_tokenized_dataset(self,
                                      data2investigate="train",
                                      column2investigate="label",
                                      num_samples=5
                                      ):
        """
        Prints a total of num_samples examples from a specified column (column2investigate)
        of a specified dataset (data2investigate).
        column2investigate : {'attention_mask', 'input_ids', 'label', 'text', 'token_type_ids'}
        data2investigate : {'train', 'val', 'test'}
        """
        # Check if the inputs are valid.
        if data2investigate not in {"train", "val", "test"}:
            raise ValueError("Invalid value for data2investigate: {}" % data2investigate)
        if column2investigate not in {'attention_mask', 'input_ids', 'label', 'text', 'token_type_ids'}:
            raise ValueError("Invalid value for column2investigate: {}" % column2investigate)

        # Print samples
        if data2investigate == "train":
            print(self.tokenized_dataset_train[column2investigate][0:num_samples])
        elif data2investigate == "val":
            print(self.tokenized_dataset_val[column2investigate][0:num_samples])
        else:
            print(self.tokenized_dataset_test[column2investigate][0:num_samples])


    def training_argument_creator(self,
                                  # output_dir='/home/kaan/code/model_training/trained',
                                  evaluation_strategy="epoch",
                                  logging_strategy="steps",
                                  save_strategy="epoch",
                                  metric_for_best_model="eval_accuracy",
                                  greater_is_better=True,
                                  load_best_model_at_end=True,
                                  # learning_rate=0.00005,
                                  # lr_scheduler_type="linear",
                                  warmup_ratio=0.0,
                                  warmup_steps=0,
                                  group_by_length=False,
                                  report_to="all",
                                  weight_decay=0.0,
                                  # per_device_train_batch_size=8
                                  save_steps=500
                                  ):
        """
        Creates training_args.
        Stores the variables in self.
        """
        # if len(self.model_name) > 100:

        self.output_dir = '/home/kaan/code/model_training/trained/translation_trick_results/translationTrickResults_{}_{}_{}epochs'.format(d1, self.model_name,self.num_training_epochs)
        print('----------- Creating training arguments -----------')
        self.training_args = TrainingArguments(output_dir=self.output_dir,
                                               evaluation_strategy=evaluation_strategy,
                                               num_train_epochs=self.num_training_epochs,
                                               logging_strategy=logging_strategy,
                                               save_strategy=save_strategy,
                                               # metric_for_best_model=metric_for_best_model,
                                               # greater_is_better=greater_is_better,
                                               load_best_model_at_end=load_best_model_at_end,
                                               # learning_rate=learning_rate,
                                               # lr_scheduler_type=lr_scheduler_type,
                                               warmup_ratio=warmup_ratio,
                                               warmup_steps=warmup_steps,
                                               group_by_length=group_by_length,
                                               report_to=report_to,
                                               weight_decay=weight_decay,
                                               save_steps=save_steps
                                               # per_device_train_batch_size=per_device_train_batch_size
                                               )
        print('----------- Created training arguments -----------')
        # print('Training arguments: {}'.format(self.training_args))

    def fine_tuning(self,
                    early_stopping_patience=50,
                    evaluation_metric="f1_macro"):

        self.evaluation_metric = evaluation_metric
        # accuracy = self.metric_accuracy
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            print('----Evaluation Predictions----')
            print(f'Predictions: {predictions}')
            print(f'Groundtruth: {labels}')
            print('------------------------------')

            if evaluation_metric == 'accuracy':
                return self.metric_accuracy.compute(predictions=predictions, references=labels)
            elif evaluation_metric == 'f1_micro':
                return self.metric_f1.compute(predictions=predictions, references=labels, average="micro")
            elif evaluation_metric == 'f1_macro':
                return self.metric_f1.compute(predictions=predictions, references=labels, average="macro")
            elif evaluation_metric == 'precision':
                return self.metric_precision.compute(predictions=predictions, references=labels, average="macro")
            elif evaluation_metric == 'recall':
                return self.metric_recall.compute(predictions=predictions, references=labels, average="macro")
            # # roc_auc_score -> needs debugging
            # elif evaluation_metric == 'roc_auc_score':
            #     return roc_auc_score(y_score=torch.softmax(logits), y_true=labels, multi_class='ovo', average='macro')


        # Initialization of the model with the weights of pre-specified model
        # Needed for hyperparameter optimization
        model_name_for_init = self.model_name
        def model_init():
            return AutoModelForSequenceClassification.from_pretrained(
                model_name_for_init, return_dict=True)

        optimizer = AdamW(self.model.parameters(),
                          lr=0.00005
                          )

        num_training_steps = self.num_training_epochs * self.len_train_data

        lr_scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=100,
            num_training_steps=num_training_steps,
            lr_end=0.00003,
        )

        self.trainer = Trainer(
            model=self.model, # Should be commented if using hyperparameter search with Ray Tune
            # model_init=model_init, # Should not be commented if using hyperparameter search with Ray Tune
            args=self.training_args,
            train_dataset=self.tokenized_dataset_train,
            eval_dataset=self.tokenized_dataset_val,
            compute_metrics=compute_metrics,
            callbacks=[MyCallback,
                       EarlyStoppingCallback(
                           early_stopping_patience=early_stopping_patience,
                       )
                       ],
            optimizers=(optimizer, lr_scheduler)
        )

        print('----------- Fine-tuning the model -----------')
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            self.trainer.train()
        else:
            raise ValueError("GPU is not available.")
        print('----------- Fine-tuned the model -----------')

    def make_prediction(self):

        def map_to_icd_codes(class_list_as_integer):
            class_list_as_string = []
            # output = {'class'}

            for class_value in class_list_as_integer:
                class_as_string = list(self.icd_code_dict.keys())[list(self.icd_code_dict.values()).index(class_value)]
                class_list_as_string.append(class_as_string)
            # class_list_as_string = [list(self.icd_code_dict.keys())[list(self.icd_code_dict.values()).index(class_value)] for class_value in class_list_as_integer]
            return class_list_as_string

        predictions = self.trainer.predict(self.tokenized_dataset_test)
        preds = np.argmax(predictions.predictions, axis=-1)

        print('----Test Predictions----')
        # print(f"Predictions untouched: {predictions}")
        print(f'Predictions: {preds}')
        print(f'Groundtruth: {predictions.label_ids}')
        print('------------------------------')

        # row_sums = torch.sum(predictions.predictions, 1)  # normalization
        # row_sums = row_sums.repeat(1, 10)  # expand to same size as out
        # y_pred_rocauc = torch.div(predictions.predictions, row_sums)  # these should be histograms

        self.test_results = {"accuracy": self.test_metric.compute(predictions=preds, references=predictions.label_ids),
                             "f1_micro": self.metric_f1.compute(predictions=preds, references=predictions.label_ids, average="micro"),
                             "f1_macro": self.metric_f1.compute(predictions=preds, references=predictions.label_ids, average="macro"),
                             "precision": self.metric_precision.compute(predictions=preds, references=predictions.label_ids, average="macro"),
                             "recall": self.metric_recall.compute(predictions=preds, references=predictions.label_ids, average="macro"),
                             "top_3_accuracy": top_k_accuracy_score(y_score=predictions.predictions, y_true=predictions.label_ids, k=3),
                             "top_5_accuracy": top_k_accuracy_score(y_score=predictions.predictions, y_true=predictions.label_ids, k=5),
                             "roc_auc_score": roc_auc_score(y_score=softmax(predictions.predictions, axis=1), y_true=predictions.label_ids, multi_class="ovo", average="macro"),
                             "confusion_matrix": confusion_matrix(y_pred=preds, y_true=predictions.label_ids),
                             "confusion_matrix_normalized": confusion_matrix(y_pred=preds, y_true=predictions.label_ids, normalize="true"),
                             "multilabel_confusion_matrix": multilabel_confusion_matrix(y_pred=preds, y_true=predictions.label_ids),
                             "classification_report": classification_report(y_pred=preds, y_true=predictions.label_ids, output_dict=True),
                             "top_2_accuracy": top_k_accuracy_score(y_score=predictions.predictions, y_true=predictions.label_ids, k=2),
                             "test_predictions":preds
                             }

        # This list is used in write_results_to_csv method
        self.test_results_as_list = ['nlp_pipeline_translation',
                                     self.output_dir, self.model_name, self.mlm_translation_direct,
                                     self.mlm_dataset, self.num_training_epochs, self.lr_scheduler_type,
                                     self.evaluation_metric,
                                     round(self.test_results["accuracy"]["accuracy"],3),
                                     round(self.test_results["f1_micro"]["f1"],3),
                                     round(self.test_results["f1_macro"]["f1"],3),
                                     round(self.test_results["precision"]["precision"],3),
                                     round(self.test_results["recall"]["recall"],3),
                                     round(self.test_results["top_3_accuracy"],3),
                                     round(self.test_results["top_5_accuracy"],3),
                                     round(self.test_results["roc_auc_score"],3),
                                     # self.test_results["confusion_matrix"],
                                     # self.test_results["multilabel_confusion_matrix"],
                                     # self.test_results["classification_report"],
                                     # self.test_results["top_2_accuracy"],
                                     '-','-','-','-',
                                     self.wandb_url,
                                     datetime.now(),
                                     self.test_results["test_predictions"]
                                     ]

        print(f'Saved under: {self.output_dir}')
        # print('###########################################################')
        # print(f"softmaxpredictions.predictions - {predictions.predictions}")
        # print('###########################################################')
        return self.test_results

    def load_from_checkpoint(self,
                             checkpoint_directory_name):
        self.loaded_tokenizer = AutoTokenizer.from_pretrained('trained/{}'.format(checkpoint_directory_name))
        self.model = AutoModelForSequenceClassification.from_pretrained('trained/{}'.format(checkpoint_directory_name))

    def predict_with_checkpoint(self,
                                path_to_raw_dataset):
        df_raw_dataset = pd.read_csv(path_to_raw_dataset, index_col=0).reset_index(drop=True)
        dataset_obj = Dataset.from_pandas(df_raw_dataset)
        tokenized_dataset = dataset_obj.map(self.tokenize_function, batched=True)

        predictions = self.model.predict(tokenized_dataset)
        preds = torch.argmax(predictions.predictions, axis=-1)
        return self.test_metric.compute(predictions=preds, references=predictions.label_ids)


    def write_results_to_csv(self):
        # Source - https://www.geeksforgeeks.org/how-to-append-a-new-row-to-an-existing-csv-file/
        with open('/home/kaan/results/results.csv', 'a', newline='') as f_object:

            writer_object = writer(f_object) # writer method is from the csv library
            writer_object.writerow(self.test_results_as_list)
            f_object.close()

    def visualize_confusion_matrix(self,
                                   save_image=False):


        display_labels = ['Z80.3', 'Z01.6', 'N64.4', 'N64.5', 'D48.6', 'Z12.3',
                          'D24', 'R92', 'Z85.3', 'C50.9']
        disp = ConfusionMatrixDisplay(confusion_matrix=self.test_results.get('confusion_matrix'),
                                      display_labels=display_labels
                                      )

        fig, ax = plt.subplots()

        # NOTE: Fill all variables here with default values of the plot_confusion_matrix
        disp.plot(include_values=True,
                  cmap='viridis',
                  ax=ax,
                  xticks_rotation=45,
                  )

        # show plot
        plt.show()

        # Save image
        if save_image:
            plt.savefig(fname=f'/home/kaan/results/confusion_matrix_images/denormalized/{self.output_dir[39:].replace("/","-")}.jpeg')

    def visualize_confusion_matrix_normalized(self,
                                              save_image=False):


        display_labels = ['Z80.3', 'Z01.6', 'N64.4', 'N64.5', 'D48.6', 'Z12.3',
                          'D24', 'R92', 'Z85.3', 'C50.9']
        disp = ConfusionMatrixDisplay(confusion_matrix=self.test_results.get('confusion_matrix_normalized'),
                                      display_labels=display_labels
                                      )

        fig, ax = plt.subplots()

        # NOTE: Fill all variables here with default values of the plot_confusion_matrix
        disp.plot(include_values=True,
                  cmap='viridis',
                  ax=ax,
                  xticks_rotation=45,
                  )

        # Save image
        if save_image:
            if len(self.output_dir) > 100:
                start_index = 65
            else:
                start_index = 39
            plt.savefig(fname=f'/home/kaan/results/confusion_matrix_images/normalized/normalized_{self.output_dir[start_index:].replace("/","-")}.jpeg')


# # Automatic
models_to_test = [
# 'dmis-lab/biobert-v1.1',
# 'emilyalsentzer/Bio_ClinicalBERT',
# 'allenai/scibert_scivocab_uncased',
# 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
# '/home/kaan/code/model_training/trained/mlm_training_results/englishModelAdaptedWithGermanCorpus_dmis-lab/biobert-v1.1_cpg-sentences_50-epochs/checkpoint-150000'
# '/home/kaan/code/model_training/trained/mlm_training_results/englishModelAdaptedWithGermanCorpus_dmis-lab/biobert-v1.1_gmw-sentences_50-epochs/checkpoint-172000'
'/home/kaan/code/model_training/trained/mlm_training_results/englishModelAdaptedWithGermanCorpus_allenai/scibert_scivocab_uncased_gmw-sentences_50-epochs/checkpoint-120000'
]
epochs_to_test = [3]

for model in models_to_test:
    for epoch in epochs_to_test:
        pipe_object = NLP_pipeline(
            model_name=model,
            num_training_epochs=epoch,
            lr_scheduler_type="polynomial",
            loaded_from_checkpoint=True,
            mlm_dataset='gmw',
            mlm_translation_direct_label='translation+mlm'
        )

        pipe_object.print_info()
        # Fine-tuning steps
        pipe_object.load_data()
        pipe_object.initialize_model_and_tokenizer()
        pipe_object.tokenize_data()
        pipe_object.training_argument_creator(
            warmup_steps=100,
            weight_decay=0.01,
            save_steps=500
        )
        pipe_object.fine_tuning(
            early_stopping_patience=50,
            evaluation_metric="f1_macro"
            # [accuracy, f1_micro, f1_macro, precision, recall, roc_auc_score (needs debugging)]
        )
        test_accuracy = pipe_object.make_prediction()
        print("----- Test set results -----")
        print(f"{test_accuracy}")
        pipe_object.write_results_to_csv()
        pipe_object.visualize_confusion_matrix(save_image=True)
        pipe_object.visualize_confusion_matrix_normalized(save_image=True)