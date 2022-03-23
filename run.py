import argparse
from argparse import ArgumentParser
from ast import parse
from email.policy import default
import os
import json
import random
from evaluation import evaluate
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from models import load_model

from Datasets import Pretrain
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', default=None, type=str)
    parser.add_argument('--method', default=None, type=str, help="Options are: baseline/kadapter/kadapter2/lora/lora2/mixreview/modular/modular_small/recadam")
    parser.add_argument('--freeze_level', default=None, type=int, help='Default value will be set to 0 for methods: baseline, mixreview, recadam, and to 1 for other methods')
    parser.add_argument('--split', default=None, type=int)
    parser.add_argument('--randomized_trial', default=None, type=int)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--data_split', default='train', type=str)
    parser.add_argument('--checkpoint_path', default=None, type=str)
    arg_ = parser.parse_args()
    if arg_.config == None:
        raise NameError("Include a config file in the argument please.")
    if arg_.freeze_level is None:
        if arg_.method in ['baseline', 'mixreview', 'recadam']:
            arg_.freeze_level = 0
        else:
            arg_.freeze_level = 1
    if arg_.seed is None:
        raise NameError('Please enter a seed')
    #Getting configurations
    with open(arg_.config) as config_file:
        hparam = json.load(config_file)


    hparam = argparse.Namespace(**hparam)

    seed = arg_.seed

    #Setting GPUs to use
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=hparam.CUDA_VISIBLE_DEVICES

    wandb_logger = None

    #Init configs that are not given
    if 'split_num' not in hparam:
        hparam.split_num = None
    if 'grad_norm' not in hparam:
        hparam.grad_norm = 0.5
    if 'weight_decay' not in hparam:
        hparam.weight_decay = 0.0
    if 'output_log' not in hparam:
        hparam.output_log = None
    if arg_.data_split == 'test':
        hparam.output_log = 'log_test'
        hparam.mode = 'test'
    if arg_.data_split != 'train' and 'evaluation' not in arg_.config:
        raise Exception('config file should be for evaluation when data split valid or test is chosen')

    #Setting configurations
    args_dict = dict(
        output_dir=hparam.output_dir, # Path to save the checkpoints
        dataset=hparam.dataset,
        dataset_version = hparam.dataset_version,
        split_num = hparam.split_num,
        split = arg_.split,
        model_name_or_path=hparam.model,
        method=arg_.method,
        freeze_level=arg_.freeze_level,
        mode=hparam.mode,
        tokenizer_name_or_path=hparam.model,
        max_input_length=hparam.input_length,
        max_output_length=hparam.output_length,
        freeze_encoder=False,
        freeze_embeds=False,
        learning_rate=hparam.learning_rate,
        weight_decay=hparam.weight_decay,
        adam_epsilon=1e-8,
        warmup_steps=0,
        train_batch_size=hparam.train_batch_size,
        eval_batch_size=hparam.train_batch_size,
        num_train_epochs=hparam.num_train_epochs,
        gradient_accumulation_steps=hparam.gradient_accumulation_steps,
        n_gpu=hparam.ngpu,
        num_workers=hparam.num_workers,
        resume_from_checkpoint=hparam.resume_from_checkpoint, 
        use_lr_scheduling = hparam.use_lr_scheduling,
        val_check_interval = 1.0,
        n_val=-1,
        n_train=-1,
        n_test=-1,
        early_stop_callback=False,
        use_deepspeed=hparam.use_deepspeed,
        opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
        max_grad_norm=hparam.grad_norm, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
        seed=arg_.seed,
        check_validation_only=hparam.check_validation,
        checkpoint_path=arg_.checkpoint_path,
        accelerator=hparam.accelerator,
        output_log=hparam.output_log,
    )
    args = argparse.Namespace(**args_dict)

    if arg_.randomized_trial:
        args.dataset_version = args.dataset_version + '_' + str(arg_.randomized_trial)
    args.output_dir = args.output_dir + '_'+ args.dataset +'_'+args.dataset_version+'_'+args.method+'_freeze_'+str(args.freeze_level)+'_seed_'+str(arg_.seed)
    args.split = arg_.split
    if args.split_num:
        args.output_dir = args.output_dir+'_split'+str(args.split)
        if args.check_validation_only:
            if args.checkpoint_path is None:
                args.checkpoint_path = os.path.join(args.output_dir.replace('split'+str(args.split), 'split'+str(args.split_num-1)), "last.ckpt")
            args.output_log = os.path.join(args.output_log, args.dataset +'_'+args.dataset_version, args.method+'_freeze_'+args.freeze_level+'_split'+str(args.split)+'.csv')
        elif args.split > 0:
            if args.checkpoint_path is None:
                args.checkpoint_path = os.path.join(args.output_dir.replace('split'+str(args.split), 'split'+str(args.split-1)), "last.ckpt")
            
    else:
        if args.check_validation_only:
            if args.checkpoint_path is None:
                args.checkpoint_path = os.path.join(args.output_dir, "last.ckpt")
            args.output_log = os.path.join(args.output_log, args.dataset +'_'+args.dataset_version, args.method+'_freeze_'+str(args.freeze_level)+'.csv')

    # Defining how to save model checkpoints during training. Details: https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.model_checkpoint.html 
    callbacks = [ModelCheckpoint(dirpath = args.output_dir, save_top_k=-1, period=1)]
    checkpoint_callback = True

    if args.output_dir=="":
        checkpoint_callback = False # Do not save model checkpoints when output dir is empty
        callbacks=[]

    # Logging Learning Rate Scheduling
    if args.use_lr_scheduling and wandb_logger:
        callbacks.append(pl.callbacks.LearningRateMonitor())

    if args.use_deepspeed:
        plugins = 'deepspeed_stage_2'
        use_fp_16 = True
    else:
        plugins = []
        use_fp_16 = False

    # Setting Flags for pytorch lightning trainer. Details: https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-flags
    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        plugins=plugins,
        gpus=args.n_gpu,
        max_epochs=args.num_train_epochs,
        precision= 16 if use_fp_16 else 32,
        amp_level=args.opt_level,
        resume_from_checkpoint=args.resume_from_checkpoint,
        gradient_clip_val=args.max_grad_norm,
        checkpoint_callback=checkpoint_callback,
        val_check_interval=args.val_check_interval,
        logger=wandb_logger,
        callbacks = callbacks,
        accelerator=args.accelerator,
    )

    #Getting the Model type & Method
    if 't5' in args.model_name_or_path:
        model_type='T5'
    elif 'gpt2' in args.model_name_or_path:
        model_type='GPT2'
    else:
        raise Exception('Select the correct model. Supporting "t5" and "gpt2" only.')
    Model = load_model(type=model_type)
    print ('Using checkpoint path: ', args.checkpoint_path)

    if args.check_validation_only:
       evaluate(args, Model)
    else:
        set_seed(seed)
        if args.checkpoint_path!="":
            model = Model.load_from_checkpoint(checkpoint_path=args.checkpoint_path, hparams=args, strict=False) 
        else:
            model = Model(args)
        trainer = pl.Trainer(**train_params)
        trainer.fit(model)
