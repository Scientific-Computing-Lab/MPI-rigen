import os
import re
import pdb
import json
import torch
import logging
import hf_data_mpi as data_mpi
from torch.optim import AdamW
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from tqdm.auto import tqdm
from torch.nn.functional import cross_entropy, one_hot
from torch.utils.data import DataLoader, Dataset
from transformers import GPTNeoXForCausalLM
from prettytable import PrettyTable
from tokenizer.tokenizer import TokompilerTokenizer, _GPT2BPETokenizer
from transformers import GPTNeoXForCausalLM, AutoModelForCausalLM, GPT2Tokenizer
from transformers import DataCollatorForLanguageModeling, get_linear_schedule_with_warmup

logger = logging.getLogger()


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'input_ids': torch.tensor(self.data[idx]['input_ids']),
                'labels': torch.tensor(self.data[idx]['labels'])}


def tokenize(args, tokenizer, sample):
    if not args.is_replaced:
        encodings = tokenizer(sample['full'], max_length=args.max_length, add_special_tokens=True, truncation=True,
                              padding=True)

        if len(encodings['input_ids']) < args.max_length:
            encodings['input_ids'].append(tokenizer.eos_token_id)
    else:
        encodings = {}
        encodings['input_ids'] = tokenizer(sample['full'], max_length=args.max_length, add_special_tokens=True,
                                           truncation=True, padding=True)
        encodings['labels'] = encodings['input_ids'][:]

    return encodings


def decode(logits, labels, tokenizer):
    pred, label = [], []
    preds = logits
    preds = preds[0]

    for token_id in preds.tolist():
        pred.append(tokenizer.decode([token_id]))

    for token_id in labels[0].tolist():
        label.append(tokenizer.decode([token_id]))

    pdb.set_trace()
    return pred, label


def test(args):
    logger.info('start test')

    # TOKENIZER
    with open(r'mpi/hf/mpi.code-snippets', 'r') as f:
        file = json.load(f)
    tokom_extended_tokens = [prefix.lower() for prefix in file.keys()] + ['parallel']

    if args.is_replaced:
        tokenizer = TokompilerTokenizer(vocab_path=args.vocab_file)
        tokenizer.add_tokens(tokom_extended_tokens)
        tokenizer.enable_padding(length=args.max_length)
    else:
        tokenizer = GPT2Tokenizer(vocab_file=args.vocab_file, merges_file=args.merge_file, padding=True,
                                  truncation=True, model_input_names=['input_ids'], padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token

    # DATA
    datasets = data_mpi.build_mpi_dataset(args)

    newd = []
    for i in range(len(datasets)):
        d = datasets[i]
        outd = d.map(lambda examples: tokenize(args, tokenizer, examples),
                     remove_columns=['program', 'code', 'mpi_labels', 'full'])
        newd.append(outd)

    traind, testd = newd

    if args.is_replaced:

        train_data = []
        for ids, labels in tqdm(zip(traind['input_ids'], traind['labels'])):
            train_data.append({'input_ids': ids, 'labels': labels})

        test_data = []
        for ids, labels in tqdm(zip(testd['input_ids'], testd['labels'])):
            test_data.append({'input_ids': ids, 'labels': labels})

        test_loader = DataLoader(dataset=CustomDataset(test_data), batch_size=1)
    else:
        collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        test_loader = DataLoader(dataset=testd, batch_size=1, collate_fn=collator)

    # MODEL
    model_name = 'mpi_poly_tokom' if args.is_replaced else 'mpi_poly_bpe_2048_3_epochs_close_1_line'
    model = AutoModelForCausalLM.from_pretrained("NinedayWang/PolyCoder-2.7B")
    # model = GPTNeoXForCausalLM.from_pretrained(os.path.join(args.save_dir, model_name))
    #
    model.to(args.device)
    model.eval()
    print('Model has been loaded')

    progress_bar = tqdm(range(len(test_loader)))

    max_new_tokens = 512
    steps = 0
    for batch_idx, batch in enumerate(test_loader):
        tensor_batch = {k: v.to(args.device) for k, v in batch.items() if
                        k in ['input_ids', 'labels', 'mask', 'attention_mask']}
        if steps == 1000:
            break
        input_ids = tensor_batch['input_ids']
        if len(input_ids[0]) < 2048:

            mask = torch.ones_like(input_ids)

            if args.is_replaced:
                mask[input_ids == tokenizer.pad_id] = 0
                mask[input_ids == tokenizer.eod] = 0
            else:
                mask[input_ids == tokenizer.eos_token_id] = 0
                mask[input_ids == tokenizer.pad_token_id] = 0

            outputs = model.generate(input_ids=input_ids, attention_mask=mask,
                                     max_new_tokens=max_new_tokens) # max_new_tokens=max(0, tokens_amount - context_size)
            logits = outputs[0]
            # labels = tensor_batch['input_ids'][0]

            pred = tokenizer.decode(logits.tolist())
            # label = tokenizer.decode(labels.tolist())

            with open(f'results/polycoder_512_close.jsonl', 'a+') as f:
                sep = ' ' if args.is_replaced else ''
                pred = sep.join(pred)
                # label = sep.join(label)
                f.write(json.dumps({'pred': pred[pred.rfind('parallel')+8:]}) + '\n')
                # f.write(json.dumps({'label': label[label.rfind('parallel')+8:]}) + '\n')
                # f.write(json.dumps({'code': label}) + '\n')

            steps += 1
            progress_bar.update(1)
    # results = ''
    # for batch_idx, batch in enumerate(test_loader):
    #     tensor_batch = {k: v.to(args.device) for k, v in batch.items() if
    #                     k in ['input_ids', 'labels', 'mask', 'attention_mask']}
    #
    #     labels = tensor_batch['input_ids'][0]
    #     labels = labels[labels != 1]
    #     # pdb.set_trace()
    #     # tensor_batch['inputs_ids'] = tensor_batch['input_ids'][0][:torch.where(tensor_batch['input_ids'][0]==tokenizer.tokenizer.encoder['parallel'])[0][0]]
    #     outputs = model(**tensor_batch)
    #     logits = outputs.logits
    #
    #     preds = torch.argmax(logits, dim=-1)
    #
    #     if args.is_replaced:
    #         preds = preds[preds != 1]
    #     else:
    #         preds = preds[preds != 50256]
    #
    #     try:
    #         pred = tokenizer.decode(preds.tolist())
    #         label = tokenizer.decode(labels.tolist())
    #         results += f'{label}\n{pred}\n\n\n'
    #         # pred_table.add_row([label[label.rfind('parallel'):] if 'parallel' in label else 'None',
    #         #                     pred[pred.rfind('parallel'):] if 'parallel' in pred else 'None'])
    #     except:
    #         print('Decoder Error')

        # progress_bar.update(1)

    # with open(f'results/{model_name}_results.log', 'w') as f:
    #     f.write(results)
