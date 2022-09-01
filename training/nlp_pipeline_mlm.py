from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers.optimization import AdamW, get_polynomial_decay_schedule_with_warmup
from datasets import load_dataset
import json
import torch

class MLM_training:
    def __init__(self,
                 base_model,
                 corpus,
                 epochs=2,
                 mlm_probability=0.15,
                 is_test=False,
                 loaded_from_checkpoint=False,
                 use_custom_name=False,
                 custom_name=None
                 # block_size=128
                 ):
        self.is_test = is_test
        self.loaded_from_checkpoint = loaded_from_checkpoint
        self.corpus = corpus
        self.epochs = epochs
        self.mlm_probability = mlm_probability
        # self.block_size = block_size

        self.dataset = None
        self.tokenized_dataset = None
        self.lm_datasets = None

        self.tokenizer = None
        self.model = AutoModelForMaskedLM.from_pretrained(base_model)
        self.model_name = base_model
        self.tokenizer_name_from_checkpoint = None
        self.use_custom_name = use_custom_name
        self.custom_name = custom_name

    def load_data(self):
        torch.cuda.empty_cache()

        if self.corpus == 'cpg':
            self.dataset = load_dataset('text',
                                        data_files={'train': '/home/kaan/scrape_data/cpg-sentences_train.txt',
                                                    'test': '/home/kaan/scrape_data/cpg-sentences_test.txt'})
        elif self.corpus == 'gmw':
            self.dataset = load_dataset('text',
                                        data_files={'train': '/home/kaan/scrape_data/gmw-sentences_train.txt',
                                                    'test': '/home/kaan/scrape_data/gmw-sentences_test.txt'})
        else:
            print('Unknown MLM dataset. Please enter the correct name - cpg or gmw')
            exit()

        self.len_train_data = self.dataset['train'].shape[0]
        # print(f'len train: {self.len_train_data}')

    def tokenize_datasets(self):

        def tokenize_function(examples):
            return tokenizer(examples["text"])

        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            block_size=128
            total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result


        if self.loaded_from_checkpoint:
            with open(f"{self.model_name}/config.json", "r") as read_file:
                data = json.load(read_file)
                self.tokenizer_name_from_checkpoint = data["_name_or_path"]
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_from_checkpoint)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.tokenized_datasets = self.dataset.map(tokenize_function, batched=True, remove_columns=["text"])

        self.lm_datasets = self.tokenized_datasets.map(
            group_texts,
            batched=True,
            batch_size=1000,
            num_proc=4,
        )

    def train_with_mlm(self,
                       save_steps=5000):

        if self.use_custom_name:
            save_path = f'/home/kaan/code/model_training/trained/mlm_training_results/{self.custom_name}_{self.model_name}_{self.corpus}-sentences_{self.epochs}-epochs'
        else:
            save_path = f'/home/kaan/code/model_training/trained/mlm_training_results/{self.model_name}_{self.corpus}-sentences_{self.epochs}-epochs'




        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

        # Data collator
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=self.mlm_probability)

        # Optimizer
        optimizer = AdamW(self.model.parameters(),
                          lr=0.00005,
                          weight_decay=0.01,
                          )

        num_training_steps = self.epochs * self.len_train_data

        lr_scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                                 num_warmup_steps=500,
                                                                 num_training_steps=num_training_steps,
                                                                 lr_end=0.00003,
                                                                 )

        training_args = TrainingArguments(
            output_dir=save_path,
            evaluation_strategy="epoch",
            # learning_rate=2e-5,
            # lr_scheduler_type="polynomial",
            weight_decay=0.01,
            num_train_epochs=self.epochs,
            logging_steps=500,
            save_steps=save_steps
            #     push_to_hub=True,
            #     push_to_hub_model_id=f"{model_name}-finetuned-mlm-cpg-sentences",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.lm_datasets["train"],
            eval_dataset=self.lm_datasets["test"],
            data_collator=data_collator,
            optimizers=(optimizer, lr_scheduler)
        )

        trainer.train()

# 'dmis-lab/biobert-v1.1',
# 'emilyalsentzer/Bio_ClinicalBERT',
# 'allenai/scibert_scivocab_uncased',

mlm_training_class = MLM_training(base_model='dmis-lab/biobert-v1.1',
                                  corpus='gmw',
                                  epochs=50,
                                  mlm_probability=0.15,
                                  is_test=False,
                                  loaded_from_checkpoint=False,
                                  use_custom_name=True,
                                  custom_name='englishModelAdaptedWithGermanCorpus')

mlm_training_class.load_data()
mlm_training_class.tokenize_datasets()
mlm_training_class.train_with_mlm(save_steps=2000)