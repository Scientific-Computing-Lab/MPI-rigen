import json
import sys
from tqdm import tqdm
from metrics import *
from transformers import GPT2Tokenizer


tokenizer = GPT2Tokenizer(vocab_file='../../../../megatron/tokenizer/gpt_vocab/gpt2-vocab.json',
                                merges_file='../../../../megatron/tokenizer/gpt_vocab/gpt2-merges.txt')


def tokenize(code, is_replaced=False):
    if is_replaced:
        return code.split()
    else:
        return tokenizer.tokenize(code)
    

def concat_vars(code):
    buf = []
    tokens = code.split()

    for idx, token in enumerate(tokens):
        if token.isnumeric():
            continue

        if token in ['var', 'arr', 'struct', 'arg'] and idx < len(tokens) - 1 and tokens[idx + 1].isnumeric():
            buf.append(f'{token}_{tokens[idx + 1]}')
        else:
            buf.append(token)

    return ' '.join(buf)


is_replaced = False
generation_file = '/home/nadavsc/shared/mpi/polycoder_300_mpi.jsonl'
print(f'Generation file: {generation_file}')
total_bleu, total_code_bleu, total_code_bert = 0, 0, 0

with open(generation_file, 'r') as f:
    for idx, line in tqdm(enumerate(f, start=1)):
        js = json.loads(line.strip())

        label = js['label']
        pred = js['pred']


        if is_replaced:
            pred = concat_vars(pred)
            label = concat_vars(label)

        total_code_bleu += calc_code_bleu(pred, label)
        print(f'CodeBleu: {total_code_bleu/idx}')







