from transformers import T5Tokenizer
from Datasets import Pretrain
from torch.utils.data import DataLoader
import csv
import os


def evaluate(args, Model):
    model = Model(args)
    if args.checkpoint_path!="":
        model = Model.load_from_checkpoint(checkpoint_path=args.checkpoint_path, hparams=args, strict=False)

    model.eval()
    model.to('cuda')
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)

    if args.mode=='pretrain' or args.mode=='finetune':
        dataset_split = 'validation'
    else:
        dataset_split = 'test'
    dataset = Pretrain(tokenizer, dataset_split, None, input_length=args.max_input_length, 
                    output_length=args.max_output_length, args=args)
    ids_to_answers = dataset.ids_to_answers

    print('Length of '+dataset_split+' data: ',len(dataset))
    loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=False)
    
    total_cnt = 0
    em_correct_num = 0
    def clean_up(text):
        text =text.replace('<pad>', '')
        text = text.replace('</s>', '')
        text = text.replace(".", '')
        text = text.replace(',', '')
        text = text.replace("'", '')
        text = text.replace('"', '')
        return text   
    # If folder doesn't exist, then create it.
    MYDIR = ("/".join((args.output_log.split('/'))[:-1]))
    CHECK_FOLDER = os.path.isdir(MYDIR)
    if not CHECK_FOLDER:
        os.makedirs(MYDIR)
        print("created folder : ", MYDIR)
    else:
        print(MYDIR, "folder already exists.")

    with open(args.output_log, 'w', newline='') as writefile:  
        writer = csv.writer(writefile)
        for batch in iter(loader):
            outs = model.model.generate(
                batch["source_ids"].cuda(),
                attention_mask=batch["source_mask"].cuda(),
                use_cache=True,
                decoder_attention_mask=batch['target_mask'].cuda(),
                max_length=args.max_output_length,
                num_beams=2,
                early_stopping=True,
            )
            dec = model.ids_to_clean_text(outs)
            texts = [tokenizer.decode(ids) for ids in batch['source_ids']]
            targets = model.ids_to_clean_text(batch['target_ids'])
                
            for i in range(len(batch['source_ids'])):
                total_cnt+=1
                lines = clean_up(texts[i])
                ground_truth = targets[i]
                predicted = dec[i]

                em = model.exact_match_score(predicted, ground_truth)  
                writer.writerow([lines, ground_truth, predicted])
                if em == 1:
                    em_correct_num += 1
    print(f'Number of total '+dataset_split+' data: {total_cnt}')

    with open(args.output_log, 'a', newline='') as writefile:  
        writer = csv.writer(writefile)
        writer.writerow([em_correct_num, em_correct_num / total_cnt])
    print(f'Number of correct predictions: {em_correct_num}. Percentage : {em_correct_num / total_cnt}')
