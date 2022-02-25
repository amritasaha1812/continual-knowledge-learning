
from torch.utils.data import Dataset
import pandas as pd
import os 
from constants import *

class Pretrain(Dataset):
    def __init__(self, tokenizer, type_path, num_samples, input_length, output_length, args, length=None):
        self.args = args
        print(f'split is {self.args.split}')
        self.tokenizer = tokenizer
        self.type_path = type_path # train/ valid/test
        self.ssm = False
        self.dataset_version = self.args.dataset_version
        if 't5' in args.model_name_or_path:
            self.model_type='T5'
        elif 'gpt2' in args.model_name_or_path:
            self.model_type='GPT2'
        if self.args.dataset == 'dah':
            data_dir = os.path.join(os.path.join(DATA_DIR, self.args.dataset), self.dataset_version+'_'+str(self.args.split))
            self.dataset = pd.read_csv(os.path.join(data_dir, type_path+'.txt'), delimiter='\t', names=['input', 'output'])
        self.input_length = input_length 
        self.output_length = output_length
        self.ids_to_answers = None
        
    def __len__(self):
        return len(self.dataset)

    def convert_to_features(self, example_batch, index=None):
        input_ = example_batch['input']
        target_ = example_batch['output']
        source = self.tokenizer.batch_encode_plus([str(input_)], max_length=self.input_length,
                                padding='max_length', truncation=True, return_tensors="pt")
        target = self.tokenizer.batch_encode_plus([str(target_)], max_length=self.output_length,
                                padding='max_length', truncation=True, return_tensors="pt")
        return source, target

    def __getitem__(self, index):
        source, target = self.convert_to_features(self.dataset.iloc[index])
        source_ids = source["input_ids"].squeeze()
        target_ids = target["input_ids"].squeeze()
        src_mask = source["attention_mask"].squeeze()
        target_mask = source["attention_mask"].squeeze()
        label_ids = -1
        ground_truth_ids = -1
        return {
                "source_ids": source_ids, 
                "source_mask": src_mask, 
                "target_ids": target_ids,
                "target_mask": target_mask,
                "label_ids": label_ids,
                "ground_truth_ids": ground_truth_ids
                }