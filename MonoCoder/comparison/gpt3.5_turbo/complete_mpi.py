from openai import OpenAI
from tqdm import tqdm
import json

start_idx = 0

with open("openai_key.txt", "r") as f:
    key = f.read()[:-1]

client = OpenAI(api_key=key)

prompt = """Generate the optimal MPI functions for the provided code, and supply in the response the entire complete code with those MPI functions:
{}
"""
#Copy the whole code then generate the optimal MPI code version of the provided code

test_file = '/home/nadavsc/LIGHTBITS/mpiricalplus/dataset/dataset_saved/bpe/2048_test/mccplus_target_dataset_gpt.jsonl'
with open(test_file, 'r') as f, open('mccplus_gpt.jsonl', 'w') as out:
    samples = f.readlines()[:1000]

    for idx, line in tqdm(enumerate(samples[start_idx:])):
        print(f'sample {idx}')
        sample = json.loads(line)

        try:
            response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt.format(sample["code"])}
            ]
            )

            output = {'program': sample["program"],
                      'code': sample["code"],
                      'label': sample["label"],
                      'pred': response.choices[0].message.content}

            out.write(json.dumps(output) + '\n')
        except Exception as e:
            print(f'failed at sample {start_idx+idx}')
