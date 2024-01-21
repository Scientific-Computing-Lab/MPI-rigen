from openai import OpenAI
from tqdm import tqdm
import numpy as np
import json
import torch
from transformers import GPT2Tokenizer, AutoModelForCausalLM


# with MPI median: 647 tokens
# without MPI median: 585 tokens
def median_tokens(test_file):
    num_tokens = []
    with open(test_file, 'r') as f:
        samples = f.readlines()

        for idx, line in enumerate(samples):
            sample = json.loads(line)
            code = sample['code']

            tokens = tokenizer.encode(code)
            num_tokens.append(len(tokens))
            print(f'{idx}: {len(tokens)} tokens')
    print(np.median(np.array(num_tokens)))


def gpt_pred(client, context_window):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt.format(context_window)}
        ]
    )

    pred = context_window + response.choices[0].message.content
    return pred


def init_model(is_gpt):
    if is_gpt:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        with open("openai_key.txt", "r") as f:
            key = f.read()[:-1]

        client = OpenAI(api_key=key)

        prompt = """Complete the following code:
        {}
        """
        return client, prompt, None, tokenizer
    vocab_file = '/home/nadavsc/LIGHTBITS/general-code-lms/code-lms/polycoder/megatron/tokenizer/gpt_vocab/gpt2-vocab.json'
    merge_file = '/home/nadavsc/LIGHTBITS/general-code-lms/code-lms/polycoder/megatron/tokenizer/gpt_vocab/gpt2-merges.txt'
    tokenizer = GPT2Tokenizer(vocab_file=vocab_file, merges_file=merge_file, padding=True,
                              truncation=True, model_input_names=['input_ids'])
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("NinedayWang/PolyCoder-2.7B")
    model.to('cuda')
    model.eval()
    return None, None, model, tokenizer


start_idx = 0
context = 11

original_mpi_codes = '/home/nadavsc/LIGHTBITS/mpiricalplus/dataset/dataset_saved/test_poly/pi.jsonl'
# no_mpi_codes = '/home/nadavsc/LIGHTBITS/mpiricalplus/dataset/dataset_saved/mccplus_target_dataset_tag.jsonl'
# median_tokens(no_mpi_codes)

is_gpt = False
client, prompt, model, tokenizer = init_model(is_gpt)

print(f'is_gpt: {is_gpt}\ncontext: {context}')
with open(original_mpi_codes, 'r') as f, open(f'pi_polycoder_context_{context}.jsonl', 'a') as out:
    samples = f.readlines()[:1000]

    for idx, line in tqdm(enumerate(samples[start_idx:])):
        print(f'sample {idx}')
        sample = json.loads(line)

        code = sample["code"]
        tokens = tokenizer.encode(code)
        tokens_amount = len(tokens)
        tokens = tokenizer.encode(code, max_length=context, truncation=True)
        context_window = tokenizer.decode(tokens)

        if is_gpt:
            pred = gpt_pred(client, context_window)
        else:
            tokens = torch.tensor(tokens).reshape([1, len(tokens)]).to('cuda')
            mask = torch.ones_like(tokens)
            mask[tokens == tokenizer.eos_token_id] = 0
            mask[tokens == tokenizer.pad_token_id] = 0
            outputs = model.generate(input_ids=tokens, attention_mask=mask, max_new_tokens=min(max(0, tokens_amount - context), 2048))
            pred = context_window + tokenizer.decode(outputs[0].tolist())

        output = {'label': code,
                  'pred': pred}

        out.write(json.dumps(output) + '\n')
        # except Exception as e:
        #     print(f'failed at sample {start_idx + idx}')
