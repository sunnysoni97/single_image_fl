import os
from pathlib import Path
from typing import Union, List
import numpy as np
from io import TextIOWrapper


def _extract_criteria(criteria: str, file: TextIOWrapper):
    file.seek(0)
    while True:
        line = file.readline()
        if (line == ""):
            break
        else:
            if (line.count(criteria) > 0):
                val = line.split(":")
                val = [item.strip() for item in val]
                val = val[-1]
                return val
    raise ValueError(f'{criteria} not found in the file!')


def _extract_server_test_acc(file: TextIOWrapper):
    keyword = "metrics_centralized"
    keyword2 = "server_test_acc"

    file.seek(0)
    last_line = ""
    while True:
        last_line = file.readline()
        if (last_line == ""):
            break
        else:
            flag = last_line.count(keyword)
            if (flag > 0):
                break

    if (last_line.count(keyword2) == 0):
        raise ValueError(f'{keyword2} not found in the file!')
    else:
        s_idx = last_line.find("{")
        e_idx = last_line.find("}")
        res_dict = last_line[s_idx:e_idx+1]
        res_dict = eval(res_dict)
        acc = res_dict[keyword2]
        acc = [val for _, val in acc]
        return acc


def _verify_same_criteria(c_values: List, n_seeds: int) -> bool:
    check = True

    for i in range(0, len(c_values), n_seeds):
        i_flag = True
        c_value = c_values[i]
        for j in range(1, n_seeds, 1):
            if (c_values[i+j] != c_value):
                i_flag = False
                break
        if (not i_flag):
            check = False
            break

    return check


def combine_seeds(n_seeds: int, dir_path: Path, criteria: Union[List, str], criteria_labels: List[str]) -> dict:

    # creating file list

    file_list = sorted(os.listdir(path=dir_path))
    for i in range(len(file_list)):
        file_list[i] = Path.joinpath(dir_path, file_list[i])

    c_values = []
    for file in file_list:
        c_value = ""
        with open(file, 'rt') as f:
            if (type(criteria) == type([])):
                for sub_criteria in criteria:
                    c_val = _extract_criteria(criteria=sub_criteria, file=f)
                    c_value += f'-{c_val}'
            elif (criteria != ""):
                c_value = _extract_criteria(criteria=criteria, file=f)
            else:
                c_value = "Generic Criteria"
        c_value = c_value.strip('-')
        c_values.append(c_value)

    if (criteria != ""):
        assert (_verify_same_criteria(c_values=c_values, n_seeds=n_seeds)
                ), f"{n_seeds} consecutive files donot have same {criteria} values!"

    if (criteria_labels):
        keys = criteria_labels
    elif (criteria != ""):
        keys = np.unique(np.array(c_values)).tolist()
    else:
        keys = [f'Line {int(i/2)}' for i in range(0, len(file_list), n_seeds)]

    acc_list = []
    for file in file_list:
        with open(file, 'rt') as f:
            acc = _extract_server_test_acc(f)
            acc_list.append(acc)

    n_sections = len(file_list)/n_seeds
    acc_list = np.split(ary=np.array(acc_list), axis=0,
                        indices_or_sections=n_sections)

    new_acc = []
    for split_view in acc_list:
        avg_acc = np.mean(a=split_view, axis=0)
        new_acc.append(avg_acc)

    data_dict = {}
    for key, line in zip(keys, new_acc):
        data_dict[key] = line

    return data_dict
