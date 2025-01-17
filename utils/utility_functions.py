import os
import sys
import pickle

import pandas as pd


def get_best_params(params_AUC_pairs, n_weeks):
    return sorted(params_AUC_pairs[:n_weeks], key=lambda x: x[1])[-1][0]


def get_safe_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir


def get_full_path(dir_name, filename, extension=None):
    if extension is not None:
        filename = str(filename) + '.' + str(extension)
    else:
        filename = str(filename)
    return os.path.join(dir_name, filename)


def force_dump_to_console(msg, copy=False):
    if copy is True:
        print(msg)  # print data to current out stream
    saved_std_out = sys.stdout
    sys.stdout = sys.__stdout__
    print(msg)
    sys.stdout = saved_std_out


def save_object_to_file(obj, filename):
    with open(filename, 'wb') as dump_file:
        pickle.dump(obj, dump_file)


def save_results_with_conf_to_csv(results, filename):
    data = results_to_2D_dict(results)
    pd.DataFrame.from_dict(data).to_csv(filename)


def save_results_to_csv(results, filename):
    data = results_to_dict(results)
    pd.DataFrame.from_dict(data).to_csv(filename)


def results_to_dict(results):
    """
    results = dict { KEY = week_num,
                     VALUE = dict{ KEY = METRIC, VALUE = VALUE   }
                    }
    """
    # define dictionary
    data = {}
    # define columns
    data['Batch'] = []
    for metric in results[next(iter(results))].keys():
        data[f'{metric}'] = []

    for batch_id in results.keys():
        data['Batch'].append(batch_id)
        for metric in results[batch_id].keys():
            data[f'{metric}'].append(results[batch_id][metric])
    return data


def results_to_2D_dict(results):
    """
    results = dict { KEY = week_num, VALUE = dict{
                                                    KEY = METRIC, VALUE = dict { KEY            :   VALUE
                                                                                 'Mean'         : mean_value
                                                                                 'Left bound'   : left_bound
                                                                                 'Right bound'  : right_bound
                                                                                }
                                                }
               }
    """
    # define dictionary
    data = {}
    # define columns
    data['Batch'] = []
    for metric in results[next(iter(results))].keys():
        data[f'{metric}_mean'] = []
        data[f'{metric}_min'] = []
        data[f'{metric}_max'] = []

    for batch_id in results.keys():
        data['Batch'].append(batch_id)
        for metric in results[batch_id].keys():
            data[f'{metric}_mean'].append(results[batch_id][metric]['Mean'])
            data[f'{metric}_min'].append(results[batch_id][metric]['Left bound'])
            data[f'{metric}_max'].append(results[batch_id][metric]['Right bound'])
    return data


def bins_to_dict(bins):
    dict_from_bins = {}
    for i in range(len(bins)):
        dict_from_bins[i] = bins[i]
    return dict_from_bins


def float_dict2string(float_dict):
    N_DIGITS = 4  # Round results to N_DIGITS digits
    res = ''
    for k, v in float_dict.items():
        res += f"{k}: {round(v, N_DIGITS)}"
    return "{{{str_}}}".format(str_=res)


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)
