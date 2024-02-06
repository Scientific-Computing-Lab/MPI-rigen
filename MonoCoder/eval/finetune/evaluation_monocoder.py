import os
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from plotly.subplots import make_subplots

from os.path import abspath, dirname

# from source.args import MPI_COMMON_CORE
MPI_COMMON_CORE = ['MPI_Finalize', 'MPI_Init', 'MPI_Comm_rank', 'MPI_Comm_size', 'MPI_Send', 'MPI_Recv', 'MPI_Send', 'MPI_Reduce', 'MPI_Bcast']
mpi_common_core_low = [func.lower() for func in MPI_COMMON_CORE]
MPI_COMMON_CORE = MPI_COMMON_CORE + mpi_common_core_low
OUTPUTS_DIR = os.path.join(dirname(dirname(abspath(__file__))), 'outputs')
DATASET = r'C:\Users\NadavSc\Desktop\projects\mpiricalplus\dataset'


def set_fname(comp700M, is_one_line):
    if comp700M:
        if is_one_line:
            return 'comp700M_1_line.jsonl'
        elif is_replaced:
            return 'comp700M_replaced_finetuned.jsonl'
        return 'comp700M.jsonl'
    return 'comp2.7B_output.txt'


def line_num_to_float(line_num):
    return round(float('.'.join(line_num.split(' '))), 2)


def tolerance_calc(loc):
    if VARIANCE == 0:
        return [loc]
    if is_one_line:
        return list(np.arange(loc- VARIANCE, loc + VARIANCE + 1))
    return list(set([round(num, 2) for num in np.concatenate((np.arange(loc - VARIANCE, loc + VARIANCE + 1), np.arange(loc - VARIANCE / 10, loc + 0.1 + VARIANCE / 10, 0.1)))]))


def count_mcc(gt_funcs, raw_data=False):
    count = {}
    for funcs in gt_funcs:
        if raw_data:
            funcs = [prefix[:-1] for prefix in re.findall(r'MPI_[\w]*[\s]*[(]', funcs)]
        for func in funcs:
            if func in MPI_COMMON_CORE:
                count[func] = count[func] + 1 if func in count else 0
    return count


def preprocess_locs_funcs(line, pattern):
    dict = {}
    funcs = re.findall(r'[(](.*?)[;]', line)
    for func in funcs:
        match = re.search(pattern, func)
        if match:
            loc, func_name = match.group().split(',')
            loc = line_num_to_float(loc)
            if is_one_line:
                while loc in dict:
                    loc += 1
            func_name = func_name[:-1].strip()
            args = func[match.span()[1]:-1].split(',')
            dict[loc] = (func_name, args, tolerance_calc(loc))
    return dict


def reorganization(df):
    gt = []
    preds = []
    if is_one_line:
        pattern = r'[0-9]+ [0-9]+,[\s]*mpi_[\w]*[\s]*[(]' if is_replaced else r'[0-9]+,[\s]*MPI_[\w]*[(]'
    else:
        pattern = r'[0-9]+ [0-9]+,[\s]*mpi_[\w]*[\s]*[(]' if is_replaced else r'[0-9]+ [0-9]+,[\s]*MPI_[\w]*[(]'

    for row in range(len(df)):
        gt.append(preprocess_locs_funcs(df.gt_text[row], pattern))
        preds.append(preprocess_locs_funcs(df.preds_text[row], pattern))
    return gt, preds


def preprocess(file_path, comp=True):
    if comp:
        with open(file_path, 'r') as f:
            gt_org = []
            preds_org = []
            for example in f:
                gen_obj = json.loads(example)
                pred = gen_obj['pred']
                label = gen_obj['label']

                pred = re.sub(r'<[|]endoftext[|]>[_]*', '(', pred)
                preds_org.append(pred)
                gt_org.append(label)
    else:
        with open(file_path, 'r') as f:
            file = f.readlines()
        gt_org = [file[idx] for idx in range(0, len(file), 5)]
        preds_org = [file[idx] for idx in range(2, len(file), 5)]
    df = pd.DataFrame({'gt_text': gt_org, 'preds_text': preds_org})
    gt, preds = reorganization(df)
    df['gt'], df['preds'] = pd.Series(gt), pd.Series(preds)
    return df


def num_funcs_selection(df, limit):
    idxs = []
    for idx, funcs in df['gt'].items():
        if len(funcs) <= limit:
            idxs.append(idx)
    return df.iloc[idxs]


def score_loc(gt, preds, common_core_only=False):
    num_locs = 0
    count_loc = 0
    for gt_line, gt_elements in gt.items():
        num_locs_flag = True
        for preds_line, preds_elements in preds.items():
            if not common_core_only or (common_core_only and gt_elements[0] in MPI_COMMON_CORE):
                if num_locs_flag:
                    num_locs += 1
                    num_locs_flag = False
            else:
                continue

            if gt_line in preds_elements[2]:
                if VERBOSE > 1:
                    print(preds_elements)
                count_loc += 1
                break
    return count_loc


def score_func(gt, preds, count, common_core_only=False):
    num_funcs = 0
    count_func = 0
    count_args = []
    for gt_line, gt_elements in gt.items():
        num_func_flag = True
        for preds_line, preds_elements in preds.items():
            if gt_line in preds_elements[2]:
                if not common_core_only or (common_core_only and gt_elements[0] in MPI_COMMON_CORE):
                    if num_func_flag:
                        num_funcs += 1
                        num_func_flag = False
                else:
                    continue

                if gt_elements[0] == preds_elements[0]:
                    count[gt_elements[0]] = count[gt_elements[0]] + 1 if gt_elements[0] in count else 0
                    count_func += 1
                    if gt_elements[0] is not None:
                        # print(f'{gt_elements[0]}: {gt_elements[1]} | {preds_elements[1]}')
                        count_args.append(len([1 for gt_arg, preds_arg in zip(gt_elements[1], preds_elements[1]) if gt_arg == preds_arg]) / len(gt_elements[1]))
                    else:
                        count_args.append(1)
                    break
    if num_funcs == 0:
        return None, None, count
    if VERBOSE > 1:
        print(count)
    return count_func/float(num_funcs), np.average(count_args), count


def score(df, common_core_only):
    count = {}
    counts_loc = []
    counts_func = []
    counts_args = []
    for idx in range(len(df)):
        if len(df['gt'].iloc[idx]) != 0:
            gt = df['gt'].iloc[idx]
            preds = df['preds'].iloc[idx]
            count_loc = score_loc(gt, preds, common_core_only)
            count_func, count_args, count = score_func(gt, preds, count, common_core_only)

            N = float(len(gt))
            counts_loc.append(count_loc / N)
            if count_func is not None:
                if count_loc != 0:
                    counts_func.append(count_func)
                    counts_args.append(count_args)
                else:
                    counts_func.append(None)
                    counts_args.append(None)
    return counts_loc, counts_func, counts_args, count


is_replaced = False
is_one_line = False
comp700M = True
VERBOSE = 1
VARIANCE = 0
FUNCTIONS_GRID = False
common_core_only = False
SAVE = {'csv': False,
        'graph': False,
        'all_graphs': True}
output_name = f'scores_var_{VARIANCE}_MCC' if common_core_only else f'scores_var_{VARIANCE}'
fname = set_fname(comp700M, is_one_line)
if VERBOSE > 0:
    print(f'Evaluating {fname}...')

if not SAVE['all_graphs']:
    file_path = os.path.join(DATASET, 'results', fname)
    df = preprocess(file_path, comp=comp700M)
    # count = count_mcc(df.gt_text, raw_data=True)

    scores = pd.DataFrame(columns=['Limit Functions', 'Locations', 'Functions', 'Args'])
    limit = df['gt'].apply(lambda x: len(x)).max() - 1 if not FUNCTIONS_GRID else 0
    while limit < df['gt'].apply(lambda x: len(x)).max():
        df_limit = num_funcs_selection(df, limit=limit+1)
        counts_loc, counts_funcs, counts_args, count = score(df_limit, common_core_only)
        if VERBOSE > 0:
            print(f'Functions Distribution: {count} \nNumber of functions: {sum(count.values())} \n\n')
        df_counts_loc = pd.DataFrame({'counts_loc': counts_loc})

        df_counts_funcs = pd.DataFrame({'counts_funcs': counts_funcs})
        df_counts_funcs.dropna(inplace=True)

        df_counts_args = pd.DataFrame({'counts_args': counts_args})
        df_counts_args.dropna(inplace=True)
        scores = pd.concat([scores, pd.DataFrame({'Limit Functions': [limit+1],
                                                  'Locations': [np.average(df_counts_loc['counts_loc'])],
                                                  'Functions': [np.average(df_counts_funcs['counts_funcs'])],
                                                  'Args': [np.average(df_counts_args['counts_args'])]}, index=[limit+1])])
        limit += 1

    if SAVE['csv']:
        with open(os.path.join(OUTPUTS_DIR, f'{output_name}_distribution.txt'), 'w') as f:
            f.write(f'Functions Distribution: {count} \nNumber of functions: {sum(count.values())} \n\n')
        scores.to_csv(os.path.join(OUTPUTS_DIR, f'{output_name}.csv'), index=False)

    if SAVE['graph']:
        plt.plot(scores['Locations'])
        plt.plot(scores['Functions'])
        plt.plot(scores['Args'])
        plt.legend(['Locations', 'Functions', 'Args'])
        plt.xlabel('#MPI Functions')
        plt.ylabel('Accuracy')
        plt.savefig(os.path.join(OUTPUTS_DIR, f'{output_name}.png'), dpi=250)
if SAVE['all_graphs']:
    output_name = 'scores_var_0-2' if not common_core_only else 'scores_var_0-2_MCC'
    title = 'Scores' if not common_core_only else 'Scores-MCC'
    csv_names = [f'scores_var_{i}.csv' if not common_core_only else f'scores_var_{i}_MCC.csv' for i in range(3)]
    dfs = [pd.read_csv(os.path.join(OUTPUTS_DIR, csv_name)) for csv_name in csv_names]
    fields = ['Locations', 'Functions', 'Args']
    dfs_figs = [pd.DataFrame({'Limit Functions': dfs[0]['Limit Functions'], 'Variance 0': dfs[0][field], 'Variance 1': dfs[1][field], 'Variance 2': dfs[2][field]}) for field in fields]

    subplots_fig = make_subplots(rows=1, cols=3, subplot_titles=['Locations', 'Functions', 'Arguments'], x_title='Limit Functions', shared_yaxes=True)
    figs = [px.bar(df[1:20], x="Limit Functions", y=["Variance 0", "Variance 1", "Variance 2"]) for df in dfs_figs]

    for fig_id, fig in enumerate(figs):
        for trace in range(len(fig['data'])):
            if fig_id < len(figs)-1:
                try:
                    fig["data"][trace].showlegend = False
                except:
                    pass
            subplots_fig.append_trace(fig["data"][trace], row=1, col=fig_id+1)
    subplots_fig.update_layout(title=title, yaxis_title='Accuracy', yaxis_range=[0, 1])
    print('Showing graph')
    subplots_fig.show()
    subplots_fig.write_image(os.path.join(OUTPUTS_DIR, f'{output_name}.png'), width=1920, height=1080)
    print('Graph saved')

# MCC in results test - All: {'MPI_Init': 1392, 'MPI_Comm_rank': 1401, 'MPI_Comm_size': 1202, 'MPI_Send': 797, 'MPI_Recv': 813, 'MPI_Finalize': 1495, 'MPI_Bcast': 271, 'MPI_Reduce': 186}
# MCC in results test (var0) - True Matches: {'MPI_Init': 608, 'MPI_Comm_rank': 387, 'MPI_Comm_size': 316, 'MPI_Finalize': 247, 'MPI_Recv': 67, 'MPI_Bcast': 3, 'MPI_Send': 48, 'MPI_Reduce': 3}