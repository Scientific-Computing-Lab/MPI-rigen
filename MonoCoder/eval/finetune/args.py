import os

MPI_COMMON_CORE = ['MPI_Finalize', 'MPI_Init', 'MPI_Comm_rank', 'MPI_Comm_size', 'MPI_Send', 'MPI_Recv', 'MPI_Send', 'MPI_Reduce', 'MPI_Bcast']
DATASET = r'/home/nadavsc/LIGHTBITS/mpiricalplus/dataset'
DATASET_SAVED = os.path.join(DATASET, 'dataset_saved')
MCCPLUS_PATH = os.path.join(DATASET_SAVED, 'mccplus_dataset_original.jsonl')
MCCPLUS_TARGET_PATH = os.path.join(DATASET_SAVED, 'mccplus_target_dataset.jsonl')