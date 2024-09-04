import os
from transformers import set_seed, TrainingArguments, HfArgumentParser, PretrainedConfig
from transformers import AutoTokenizer
from datasets import load_dataset
import torch
import wandb
from args import ModelArguments, DatasetArguments
from model import MSSEModel
from trainer import MyTrainer
from mteb import MTEB
from prettytable import PrettyTable
from config import MSSEConfig
import logging
import pandas as pd
import random
import torch.optim as optim
import pandas as pd
from datasets import Dataset

logger = logging.getLogger(__name__)


def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids


def eval_mteb(model, batch_size):
    tasks = [
        "STS12",
        "STS13",
        "STS14",
        "STS15",
        "STS16",
        "STSBenchmark",
        "SICK-R",
    ]
    evaluation = MTEB(tasks=tasks, task_langs=["en"], task_categories=['S2S'])
    results = evaluation.run(model, overwrite_results=True, batch_size=batch_size, eval_splits=['test'], output_folder='mteb_results/'+wandb.run.name)
    sts12 = results['STS12']['test']['cos_sim']['spearman']
    sts13 = results['STS13']['test']['cos_sim']['spearman']
    sts14 = results['STS14']['test']['cos_sim']['spearman']
    sts15 = results['STS15']['test']['cos_sim']['spearman']
    sts16 = results['STS16']['test']['cos_sim']['spearman']
    sickr = results['SICK-R']['test']['cos_sim']['spearman']
    stsb = results['STSBenchmark']['test']['cos_sim']['spearman']
    avg_sts = (sts12 + sts13 + sts14 + sts15 + sts16 + sickr + stsb) / 7
    wandb.summary['STS12'] = sts12
    wandb.summary['STS13'] = sts13
    wandb.summary['STS14'] = sts14
    wandb.summary['STS15'] = sts15
    wandb.summary['STS16'] = sts16
    wandb.summary['SICK-R'] = sickr
    wandb.summary['STSBenchmark'] = stsb
    wandb.summary['mteb_avg_sts'] = avg_sts
    return results

    
# For few-shot sentence embeddings to obtain fixed ratio of the dataset.
def split_train_dataset(datasets, split_ratio=0.1, random_seed=None):
    random.seed(random_seed)

    train_dataset_length = len(datasets["train"])

    shuffled_indices = list(range(train_dataset_length))
    random.shuffle(shuffled_indices)

    num_samples = int(split_ratio * train_dataset_length) 
    random_indices = shuffled_indices[:num_samples]

    train_dataset_split = datasets["train"].select(random_indices)
    return train_dataset_split


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DatasetArguments))
    model_args, training_args, dataset_args = parser.parse_args_into_dataclasses()
    
    wandb.init(project='MSSE-49', name=model_args.output_dir_name, dir=model_args.output_dir_name)
    set_seed(training_args.seed)
    wandb.config.update(model_args)
    wandb.config.update(training_args)
    wandb.config.update(dataset_args)
    training_args.output_dir = 'results/' + wandb.run.name
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    config = MSSEConfig(
        encoder_name_or_path=model_args.model_name_or_path,
        max_length=model_args.max_length,
        decoder_num_heads=model_args.decoder_num_heads,
        decoder_num_layers=model_args.decoder_num_layers,
        decoder_noise_dropout=model_args.decoder_target_dropout,
        pooler=model_args.pooler,
        do_contrastive=model_args.do_contrastive,
        do_generative=model_args.do_generative,
        prompt_format=model_args.prompt_format,
        contrastive_weight=model_args.contrastive_weight,
        generative_weight=model_args.generative_weight,
        contrastive_temp=model_args.contrastive_temp,

    )
    print(config)

    model = MSSEModel(config)
    
    def map_fn(example):

        max_length = model_args.max_length
        if config.pooler == 'mask':

            prompt_len = len(tokenizer(config.prompt_format, add_special_tokens=False)['input_ids'])
            def preprocess_sentence(sentence):
                tokenized_sentence = tokenizer.decode(
                    tokenizer(sentence, padding=True, truncation=True, max_length=config.max_length)['input_ids'],
                    skip_special_tokens=True
                )
                return config.prompt_format.replace('[X]', tokenized_sentence).replace('[MASK]', tokenizer.mask_token)
        
            example['sent0'] = preprocess_sentence(example['sent0'])
            example['sent1'] = preprocess_sentence(example['sent1'])
            
            if 'hard_neg' in example:
                example['hard_neg'] = preprocess_sentence(example['hard_neg'])

            max_length = max_length + prompt_len

        original_inputs = tokenizer(example['sent0'], padding='max_length', truncation=True, max_length=max_length)
        example['input_ids'] = original_inputs['input_ids']
        example['attention_mask'] = original_inputs['attention_mask']
        
        positive_inputs = tokenizer(example['sent1'], padding='max_length', truncation=True, max_length=max_length)
        example['positive_input_ids'] = positive_inputs['input_ids']
        example['positive_attention_mask'] = positive_inputs['attention_mask']

        if 'hard_neg' in example:
            negative_inputs = tokenizer(example['hard_neg'], padding='max_length', truncation=True, max_length=max_length)
            example['negative_input_ids'] = negative_inputs['input_ids']
            example['negative_attention_mask'] = negative_inputs['attention_mask']
        return example
    

    def load_and_clean_csv(file_path):
        try:
            df = pd.read_csv(file_path, delimiter="\t" if "tsv" in file_path else ",", encoding='utf-8')
        except UnicodeDecodeError:
            print("UTF-8 decoding failed. Trying latin1...")
            df = pd.read_csv(file_path, delimiter="\t" if "tsv" in file_path else ",", encoding='latin1')
        df = df.applymap(lambda x: x if isinstance(x, str) else str(x))
        return Dataset.from_pandas(df)
    
    if dataset_args.train_dataset == "data/output.csv":
        dataset = load_and_clean_csv(dataset_args.train_dataset)
    else:
        raise NotImplementedError()
    
    
    dataset = dataset.map(
        map_fn, 
        batched=False, 
        num_proc=12,
        # remove_columns=column_names,
        load_from_cache_file=True,
        ).train_test_split(0.1, seed=training_args.seed, shuffle=True)
    test_valid = dataset['test'].train_test_split(0.01)

    
    trainer = MyTrainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            # train_dataset=dataset,
            train_dataset=dataset['train'],
            eval_dataset=test_valid['test'],
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

    trainer.train()
    mteb_results = eval_mteb(model, batch_size=training_args.eval_batch_size)
    table = PrettyTable(["Name", "Value"])

    # Add rows
    table.add_row(["STS12", wandb.summary['STS12']])
    table.add_row(["STS13", wandb.summary['STS13']])
    table.add_row(["STS14", wandb.summary['STS14']])
    table.add_row(["STS15", wandb.summary['STS15']])
    table.add_row(["STS16", wandb.summary['STS16']])
    table.add_row(["SICK-R", wandb.summary['SICK-R']])
    table.add_row(["STSBenchmark", wandb.summary['STSBenchmark']])
    table.add_row(["Avg.", wandb.summary['mteb_avg_sts']])
    # Print the table
    print(table)
    
    wandb.finish()
