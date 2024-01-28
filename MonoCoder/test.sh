#!/bin/bash

python main.py      --do_test                                                           \
		    --task mpi                                                          \
                    --models_dir ../models				                \
		    --model_name monocoder_700M                                         \
		    --max_length 2048                                                   \
                    --device cuda                                                       \
                    --tokenizer_type GPT2BPETokenizer                                   \
                    --logger mpi_bpe_orig.log                                           \
		    --data_path ../data/test
