#!/bin/bash

python main.py      --do_finetune                                                       \
		    --task mpi                                                          \
                    --models_dir /home/nadavsc/shared/models/hf_checkpoints             \
		    --model_name allc_gpt2tok_700M                                      \
                    --batch_size 2                                                      \
		    --max_length 2048                                                   \
                    --num_epochs 3                                                      \
                    --device cuda                                                       \
                    --tokenizer_type GPT2BPETokenizer                                   \
                    --logger mpi_bpe_orig.log                                           \
		    --data_path /home/nadavsc/LIGHTBITS/mpiricalplus/dataset/dataset_saved/bpe/2048_train

