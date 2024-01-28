import os
import sys

sys.path.append('tokenizer')
from hf_data_mpi import dataset_preprocess
import argparse
import finetune_mpi
import test_mpi
import logging
from prettytable import PrettyTable


def main(args):
    if args.do_finetune:
        finetune_mpi.finetune(args)

    if args.do_test:
        test_mpi.test(args)

    if args.do_preprocess:
        dataset_preprocess(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.register('type', 'bool', lambda v: v.lower() in ['yes', 'true', 't', '1', 'y'])

    # Model arguments
    parser.add_argument('--models_dir', help='Specify the directory for models')
    parser.add_argument('--model_name', help='Specify the model name')
    parser.add_argument('--do_finetune', action='store_true', help='Whether to finetune')
    parser.add_argument('--do_eval', action='store_true', help='Whether to evaluation')
    parser.add_argument('--do_test', action='store_true', help='Whether to test')
    parser.add_argument('--do_preprocess', action='store_true', help='Filter examples to the max_length limit')
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'], help='Specify the device (cpu or cuda)')
    parser.add_argument('--logger', default='info.log', help='Set logger file name')
    parser.add_argument('--max_length', type=int, default=2048, help='Max length of an example')

    # Data arguments
    parser.add_argument('-t', '--tokenizer_type', type=str,
                        choices=['GPT2BPETokenizer', 'Tokompiler', 'HFGPT2Tokenizer'], default='HFGPT2Tokenizer')
    parser.add_argument('-v', '--vocab_file', type=str, default='tokenizer/gpt_vocab/gpt2-vocab.json')
    parser.add_argument('-m', '--merge_file', type=str, default='tokenizer/gpt_vocab/gpt2-merges.txt')
    parser.add_argument('-d', '--data_path', type=str, default=f'data/train')
    parser.add_argument('--data_device', default='cpu', choices=['cpu', 'gpu', 'mixed'])
    parser.add_argument('--is_replaced', default=False, action='store_true')
    parser.add_argument('--save', type=bool, default=True)

    # The following arguments are leftover from megatron settings -- you can keep the defaults
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--make_vocab_size_divisible_by', type=int, default=128)
    parser.add_argument('--model_parallel_size', type=int, default=1)

    # Training args
    parser.add_argument('--save_dir', type=str, default='outputs', help="Directory to save model checkpoints")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Big batch sizes are allowed (total #tokens per batch 262144)")
    parser.add_argument('--lr', type=float, default=16e-5, help="Learning rate for the optimizer")
    parser.add_argument('--warmup_steps', type=int, default=400, help="Number of warmup steps")
    parser.add_argument('--weight_decay', type=float, default=0, help="Weight decay for the optimizer")
    parser.add_argument('--training_steps', type=int, default=100, help="Total number of training steps")
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--adam_beta1', type=float, default=0.9, help="Beta1 for the Adam optimizer")
    parser.add_argument('--adam_beta2', type=float, default=0.999, help="Beta2 for the Adam optimizer")
    parser.add_argument('--adam_eps', type=float, default=1e-8, help="Epsilon for the Adam optimizer")
    parser.add_argument('--freeze', action='store_true', help="freeze the first layers of the model")

    main_args = parser.parse_args()

    # Define a logger
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)

    console = logging.StreamHandler()
    console.setLevel(level=logging.INFO)
    logger.addHandler(console)

    file = logging.FileHandler(os.path.join(main_args.logger))
    file.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s | %(filename)s | line %(lineno)d] - %(levelname)s: %(message)s')
    file.setFormatter(formatter)
    logger.addHandler(file)

    # Logging configuration
    config_table = PrettyTable()
    config_table.field_names = ["Configuration", "Value"]
    config_table.align["Configuration"] = "l"
    config_table.align["Value"] = "l"
    for config, value in vars(main_args).items():
        config_table.add_row([config, str(value)])
    logger.debug('Configurations:\n{}'.format(config_table))

    if main_args.is_replaced:
        main_args.vocab_file = r'tokenizer/tokompiler/tokenizer_vocab/vocab.txt'

    main(main_args)