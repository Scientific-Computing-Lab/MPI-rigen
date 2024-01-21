import pdb
import json
import glob
import os, sys

import datasets as trd
from datasets import Sequence, Value
import numpy
import torch
from transformers import GPTNeoXForCausalLM, GPT2Tokenizer
from tokenizer.tokenizer import TokompilerTokenizer, build_tokenizer

# sys.path.append('/home/nadavsc/LIGHTBITS/code-lms/polycoder/tasks/tokenizer')
with open(r'/home/nadavsc/LIGHTBITS/mpiricalplus/source/dataset/mpi.code-snippets', 'r') as f:
    file = json.load(f)
tokom_extended_tokens = [prefix.lower() for prefix in file.keys()] + ['parallel']


def dataset_preprocess(args):
    def tokenize(args, tokenizer, sample):
        if not args.is_replaced:
            encodings = tokenizer(sample, add_special_tokens=True,
                                  truncation=False, padding=True)

            if len(encodings['input_ids']) < args.max_length:
                encodings['input_ids'].append(tokenizer.eos_token_id)
        else:
            encodings = {}
            encodings['input_ids'] = tokenizer(sample, add_special_tokens=True,
                                               truncation=False, padding=False)
            encodings['labels'] = encodings['input_ids'][:]
        return encodings

    if args.is_replaced:
        tokenizer = TokompilerTokenizer(vocab_path=args.vocab_file)
        tokenizer.add_tokens(tokom_extended_tokens)
    else:
        # tokenizer = AutoTokenizer.from_pretrained("NinedayWang/PolyCoder-2.7B",
        #                                           truncation=True, model_input_names=['input_ids'])

        tokenizer = GPT2Tokenizer(vocab_file=args.vocab_file, merges_file=args.merge_file, padding=False,
                                truncation=False, model_input_names=['input_ids'])
        tokenizer.pad_token = tokenizer.eos_token

    count = 0
    encoder = 'tokom' if args.tokenizer_type=='Tokompiler' else 'bpe'
    dataset_dir = args.data_path
    dpath = f'{dataset_dir}/mccplus_target_dataset_1_line.jsonl'
    with open(dpath, 'r') as db:
        with open(os.path.join(dataset_dir, f'mccplus_target_dataset_1_line_{args.max_length}_{encoder}.jsonl'), 'w') as target_db:
            for program in db:
                json_obj = json.loads(program)
                if args.is_replaced:
                    sep_token = '[SEP]'
                    eos_token = '[EOS]'
                else:
                    sep_token = '\n'
                    eos_token = ''  # eos equals to padding

                full = f'{json_obj["code"]} {sep_token} parallel {json_obj["mpi_labels"]} {eos_token}'
                full_encoded = tokenize(args, tokenizer, full)
                if len(full_encoded['input_ids']) <= args.max_length:
                    count += 1
                    target_db.write(program)
                    print(f'{count} examples in dataset')


def build_mpi_dataset(args, rebuild=False):
    if (not rebuild) and os.path.exists(os.path.join(args.data_path, "dataset_dict.json")):
        print('Loading dataset')
        tokenized_dataset = trd.load_from_disk(args.data_path)
    else:
        # Build the dataset
        print('Building dataset')
        tokenizer = build_tokenizer(args)

        if args.tokenizer_type.lower() == 'GPT2BPETokenizer'.lower():
            eos_token = tokenizer.eod_id

        elif args.tokenizer_type.lower() == 'Tokompiler'.lower():
            eos_token = tokenizer.eod
            tokenizer.tokenizer.add_tokens(tokom_extended_tokens)

        else:
            raise NotImplementedError(f'{args.tokenizer_type} tokenizer type is not supported')


        feature_types = trd.Features({
            "program": Value("string"),
            "code": Value("string"),
            "mpi_labels": Value("string"),
        })

        dataset_dir = args.data_path
        dpath = glob.glob(f'{dataset_dir}/*target*.jsonl')
        d = trd.load_dataset('json', data_files=dpath, features=feature_types, split=['train[0%:90%]', 'train[90%:100%]'])
        d = trd.DatasetDict({'train': d[0], 'test': d[1]})

        def tokenize_and_parse(example, eos_token=eos_token):
            code = example["code"]
            mpi_labels = example["mpi_labels"]
            if args.is_replaced:
                sep_token = '[SEP]'
                eos_token = '[EOS]'
            else:
                sep_token = '\n'
                eos_token = '' # eos equals to padding

            if args.do_test:
                example["full"] = f'{code} {sep_token} parallel '
            else:
                example["full"] = f'{code} {sep_token} parallel {mpi_labels} {eos_token}'
            return example


        # JSON fields are:
        #   program: a string identifier
        #   code: text of the source code, with each line numbered
        #   mpi_labels: the (location, mpi_function) tuples to predict as outputs

        tokenized_dataset = d.map(tokenize_and_parse, batched=False)
        tokenized_dataset.set_format(output_all_columns=True)

        tokenized_dataset.save_to_disk(args.data_path)

    return tokenized_dataset["train"], tokenized_dataset["test"]
