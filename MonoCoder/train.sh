#!/bin/bash

python main.py      --do_finetune                                                       \
		    --task mpi                                                          \
                    --models_dir ../models					        \
		    --model_name monocoder_700M                                         \
                    --batch_size 2                                                      \
		    --max_length 2048                                                   \
                    --num_epochs 3                                                      \
                    --device cuda                                                       \
                    --tokenizer_type GPT2BPETokenizer                                   \
                    --logger mpi_bpe_orig.log                                           \
		    --data_path ../data/train

