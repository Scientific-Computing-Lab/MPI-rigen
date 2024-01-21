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
OUTPUTS_DIR = os.path.join(dirname(dirname(abspath(__file__))), 'outputs')
DATASET = r'/home/nadavsc/LIGHTBITS/mpiricalplus/dataset'

def line_num_to_float(line_num):
    return round(float('.'.join(line_num.split(' '))), 2)


def loc_extract(line):
    return [line_num_to_float(line_num) for line_num in re.findall(r'[0-9]+ [0-9]+', line)]


def tolerance_calc(loc):
    if VARIANCE == 0:
        return [loc]
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


def export(line, func_export=True):
    arr = []
    funcs = re.findall(r'[(](.*?)[;]', line)
    for func in funcs:
        match = re.search(r'MPI_[\w]*[(]', func)
        if match:
            arr.append(match.group()[:-1]) if func_export else arr.append(func[match.span()[1]:-1].split(','))
        else:
            arr.append(None)
    return arr


def preprocess_locations(df):
    df['gt_locations'] = df['gt_text'].apply(lambda x: loc_extract(x))
    df['preds_locations'] = df['preds_text'].apply(lambda x: loc_extract(x))


def preprocess_functions(df):
    df['gt_funcs'], df['preds_funcs'], df['gt_args'], df['preds_args'], df['gt'] = '', '', '', '', ''
    for idx in range(len(df)):
        df['gt_funcs'][idx] = export(df['gt_text'][idx], func_export=True)
        df['preds_funcs'][idx] = export(df['preds_text'][idx], func_export=True)
        df['gt_args'][idx] = export(df['gt_text'][idx], func_export=False)
        df['preds_args'][idx] = export(df['preds_text'][idx], func_export=False)


def preprocess_locs_funcs(line):
    dict = {}
    funcs = re.findall(r'[(](.*?)[;]', line)
    for func in funcs:
        match = re.search(r'[0-9]+ [0-9]+,[\s]*MPI_[\w]*[(]', func)
        if match:
            loc, func_name = match.group().split(',')
            loc = line_num_to_float(loc)
            func_name = func_name[:-1].strip()
            args = func[match.span()[1]:-1].split(',')
            dict[loc] = (func_name, args, tolerance_calc(loc))
    return dict


def reorganization(df):
    gt = []
    preds = []
    for row in range(len(df)):
        gt.append(preprocess_locs_funcs(df.gt_text[row]))
        preds.append(preprocess_locs_funcs(df.preds_text[row]))
    return gt, preds


def preprocess(file_path):
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
                count_loc += 1
                break
    return count_loc


def score_func(gt, preds, count_loc_dist, count, common_core_only=False):
    num_funcs = 0
    count_func = 0
    count_args = []
    for gt_line, gt_elements in gt.items():
        num_func_flag = True
        for preds_line, preds_elements in preds.items():
            if gt_line in preds_elements[2]:
                if not common_core_only or (common_core_only and gt_elements[0] in MPI_COMMON_CORE):
                    if num_func_flag:
                        count_loc_dist[gt_elements[0]] = count_loc_dist[gt_elements[0]] + 1 if gt_elements[0] in count_loc_dist else 1
                        num_funcs += 1
                        num_func_flag = False
                else:
                    continue

                if gt_elements[0] == preds_elements[0]:
                    count[gt_elements[0]] = count[gt_elements[0]] + 1 if gt_elements[0] in count else 1
                    count_func += 1
                    if gt_elements[0] is not None:
                        # print(f'{gt_elements[0]}: {gt_elements[1]} | {preds_elements[1]}')
                        count_args.append(len([1 for gt_arg, preds_arg in zip(gt_elements[1], preds_elements[1]) if gt_arg == preds_arg]) / len(gt_elements[1]))
                    else:
                        count_args.append(1)
                    break
    if num_funcs == 0:
        return None, None, count_loc_dist, count
    print(count)
    return count_func/float(num_funcs), np.average(count_args), count_loc_dist, count


def score(df, common_core_only):
    count = {}
    count_loc_dist = {}
    counts_loc = []
    counts_func = []
    counts_args = []
    for idx in range(len(df)):
        gt = df['gt'].iloc[idx]
        preds = df['preds'].iloc[idx]
        count_loc = score_loc(gt, preds, common_core_only)
        count_func, count_args, count_loc_dist, count = score_func(gt, preds, count_loc_dist, count, common_core_only)

        N = float(len(gt))
        counts_loc.append(count_loc / N)
        if count_func is not None:
            if count_loc != 0:
                counts_func.append(count_func)
                counts_args.append(count_args)
            else:
                counts_func.append(None)
                counts_args.append(None)
    return counts_loc, counts_func, counts_args, count_loc_dist, count


VARIANCE = 2
FUNCTIONS_GRID = True
common_core_only = False
SAVE = {'csv': False,
        'graph': False,
        'all_graphs': True}

file_path = os.path.join(DATASET, 'results', 'bpe_1000_examples_close.jsonl')
df = preprocess(file_path)
# count = count_mcc(df.gt_text, raw_data=True)

scores = pd.DataFrame(columns=['Limit Functions', 'Locations', 'Functions', 'Args'])
funcs_dist_gt = pd.DataFrame()
funcs_dist_pred = pd.DataFrame()
limit = df['gt'].apply(lambda x: len(x)).max() - 1 if not FUNCTIONS_GRID else 0
while limit < df['gt'].apply(lambda x: len(x)).max():
    df_limit = num_funcs_selection(df, limit=limit+1)
    counts_loc, counts_funcs, counts_args, count_loc_dist, count = score(df_limit, common_core_only)
    df_counts_loc = pd.DataFrame({'counts_loc': counts_loc})

    df_counts_funcs = pd.DataFrame({'counts_funcs': counts_funcs})
    df_counts_funcs.dropna(inplace=True)

    df_counts_args = pd.DataFrame({'counts_args': counts_args})
    df_counts_args.dropna(inplace=True)
    funcs_dist_gt = pd.concat([funcs_dist_gt, pd.DataFrame(count_loc_dist, index=[limit+1])])
    funcs_dist_pred = pd.concat([funcs_dist_pred, pd.DataFrame(count, index=[limit+1])])
    scores = pd.concat([scores, pd.DataFrame({'Limit Functions': [limit+1],
                                              'Locations': [np.average(df_counts_loc['counts_loc'])],
                                              'Functions': [np.average(df_counts_funcs['counts_funcs'])],
                                              'Args': [np.average(df_counts_args['counts_args'])]}, index=[limit+1])])
    limit += 1

name = f'scores_var_{VARIANCE}_MCC' if common_core_only else f'scores_var_{VARIANCE}'
if SAVE['csv']:
    funcs_dist_gt.to_csv(os.path.join(OUTPUTS_DIR, f'{name}_funcs_dist_gt.csv'), index=False)
    funcs_dist_pred.to_csv(os.path.join(OUTPUTS_DIR, f'{name}_funcs_dist_pred.csv'))
    scores.to_csv(os.path.join(OUTPUTS_DIR, f'{name}.csv'), index=False)
if SAVE['graph']:
    plt.plot(scores['Locations'])
    plt.plot(scores['Functions'])
    plt.plot(scores['Args'])
    plt.legend(['Locations', 'Functions', 'Args'])
    plt.xlabel('#MPI Functions')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(OUTPUTS_DIR, f'{name}.png'), dpi=250)
if SAVE['all_graphs']:
    name = 'scores_var_0-2' if not common_core_only else 'scores_var_0-2_MCC'
    title = 'Scores' if not common_core_only else 'Scores-MCC'
    csv_names = [f'scores_var_{i}.csv' if not common_core_only else f'scores_var_{i}_MCC.csv' for i in range(3)]
    dfs = [pd.read_csv(os.path.join(OUTPUTS_DIR, csv_name)) for csv_name in csv_names]
    fields = ['Locations', 'Functions', 'Args']
    dfs_figs = [pd.DataFrame({'Limit Functions': dfs[0]['Limit Functions'], 'Variance 0': dfs[0][field], 'Variance 1': dfs[1][field], 'Variance 2': dfs[2][field]}) for field in fields]

    subplots_fig = make_subplots(rows=1, cols=3, subplot_titles=['Locations', 'Functions', 'Arguments'], x_title='Limit Functions')
    figs = [px.bar(df, x="Limit Functions", y=["Variance 0", "Variance 1", "Variance 2"]) for df in dfs_figs]

    for fig_id, fig in enumerate(figs):
        for trace in range(len(fig['data'])):
            if fig_id < len(figs)-1:
                try:
                    fig["data"][trace].showlegend = False
                except:
                    pass
            subplots_fig.append_trace(fig["data"][trace], row=1, col=fig_id+1)
    subplots_fig.update_layout(title=title, yaxis_title='Accuracy')
    subplots_fig.show()
    subplots_fig.write_image(fr'/home/nadavsc/LIGHTBITS/mpiricalplus/{name}.png', width=1920, height=1080)

# MCC in results test - All: {'MPI_Init': 1392, 'MPI_Comm_rank': 1401, 'MPI_Comm_size': 1202, 'MPI_Send': 797, 'MPI_Recv': 813, 'MPI_Finalize': 1495, 'MPI_Bcast': 271, 'MPI_Reduce': 186}
# MCC in results test (var0) - True Matches: {'MPI_Init': 608, 'MPI_Comm_rank': 387, 'MPI_Comm_size': 316, 'MPI_Finalize': 247, 'MPI_Recv': 67, 'MPI_Bcast': 3, 'MPI_Send': 48, 'MPI_Reduce': 3}