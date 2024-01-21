import json
import os
import torch
import logging
import hf_data_mpi as data_mpi
from torch.optim import AdamW
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup, Trainer, TrainingArguments
from tqdm.auto import tqdm
from torch.nn.functional import cross_entropy, one_hot
from torch.utils.data import DataLoader, Dataset
from transformers import GPTNeoXForCausalLM, GPT2Tokenizer
from transformers import DataCollatorForLanguageModeling, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForCausalLM
from tokenizer.tokenizer import TokompilerTokenizer

from torch.cuda.amp import autocast

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
        encodings = tokenizer(sample['full'], max_length=args.max_length, add_special_tokens=True,
                              truncation=True, padding=True)

        if len(encodings['input_ids']) < args.max_length:
            encodings['input_ids'].append(tokenizer.eos_token_id)
    else:
        encodings = {}
        encodings['input_ids'] = tokenizer(sample['full'], max_length=args.max_length, add_special_tokens=True,
                                           truncation=True, padding=True)
        encodings['labels'] = encodings['input_ids'][:]

    return encodings


def finetune(args):
    logger.info(f'start finetune {args.model_name}')

    # TOKENIZER
    with open(r'mpi/hf/mpi.code-snippets', 'r') as f:
        file = json.load(f)
    tokom_extended_tokens = [prefix.lower() for prefix in file.keys()] + ['parallel']

    if args.is_replaced:
        tokenizer = TokompilerTokenizer(vocab_path=args.vocab_file)
        tokenizer.add_tokens(tokom_extended_tokens)
        tokenizer.enable_padding(length=args.max_length)
    else:
        # tokenizer = AutoTokenizer.from_pretrained("NinedayWang/PolyCoder-2.7B",
        #                                           truncation=True, model_input_names=['input_ids'])

        tokenizer = GPT2Tokenizer(vocab_file=args.vocab_file, merges_file=args.merge_file, padding=True,
                                truncation=True, model_input_names=['input_ids'])
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
    import pdb
    pdb.set_trace()

    if args.is_replaced:

        train_data = []
        for ids, labels in tqdm(zip(traind['input_ids'], traind['labels'])):
            train_data.append({'input_ids': ids, 'labels': labels})

        test_data = []
        for ids, labels in tqdm(zip(testd['input_ids'], testd['labels'])):
            test_data.append({'input_ids': ids, 'labels': labels})

        train_loader = DataLoader(dataset=CustomDataset(train_data), batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=CustomDataset(test_data), batch_size=args.batch_size)
    else:
        collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        train_loader = DataLoader(dataset=traind, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
        test_loader = DataLoader(dataset=testd, batch_size=args.batch_size, collate_fn=collator)

    # MODEL
    # model = AutoModelForCausalLM.from_pretrained("NinedayWang/PolyCoder-2.7B")  # , torch_dtype=torch.float16)
    # model.half()
    model = GPTNeoXForCausalLM.from_pretrained(os.path.join(args.models_dir, args.model_name))
    output_model_name = 'mpi_poly_tokom' if args.is_replaced else 'mpi_poly_bpe'
    model.train()
    print('model has been loaded')

    # update model embeddings
    if args.is_replaced:
        embedding_layer = model.get_input_embeddings()
        num_embeddings = embedding_layer.weight.shape[0]
        new_num_embeddings = num_embeddings + len(tokom_extended_tokens)
        model.resize_token_embeddings(new_num_embeddings)
        logger.info(f'Embedding layer has changed: {num_embeddings} -> {new_num_embeddings}')

    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_eps,
                      weight_decay=args.weight_decay)

    lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                   num_warmup_steps=100,
                                                   num_training_steps=(len(train_loader) * args.num_epochs), )

    model.to(args.device)

    for name, param in model.named_parameters():
        print(f"Parameter: {name}, Type: {param.dtype}")
    # TRAIN
    # best_loss = 4
    # best_model_state_dict = None
    for epoch in range(args.num_epochs):
        pbar = tqdm(train_loader, miniters=2, desc=f"Epoch {epoch}")
        loss_total = 0.0

        for step, batch in enumerate(train_loader):
            tensor_batch = {k: v.to(args.device) for k, v in batch.items() if
                            k in ['input_ids', 'labels', 'mask', 'attention_mask']}

            outputs = model(**tensor_batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            loss_total += loss.detach().clone().item()
            cur_loss = loss_total / (step + 1)

            if (step > 0) and (step % 10 == 0):
                logger.info(f'epoch {epoch} loss: {cur_loss}')
                pbar.set_postfix({"avg_train_loss": loss_total / step})

            # if cur_loss < best_loss:
            #     best_model_state_dict = model.state_dict()
            #     best_loss = cur_loss
            pbar.update(1)

        # VALIDATION
        # val_loss = 0.0
        # for step_val, batch_val in enumerate(test_loader):
        #     tensor_batch = {k: v.to(args.device) for k, v in batch_val.items() if k in ['input_ids', 'labels', 'mask', 'attention_mask']}

        #     outputs = model(**tensor_batch)
        #     loss = outputs.loss
        #     val_loss += loss.detach().clone().item()
        # logger.info(f'val loss:  {val_loss / (step_val+1)}')

        print('save model')
        model.save_pretrained(os.path.join(args.save_dir, output_model_name), from_pt=True)
    # logger.info(f'Best loss: {best_loss}')
    # model.load_state_dict(best_model_state_dict)
    model.save_pretrained(os.path.join(args.save_dir, output_model_name), from_pt=True)
