import argparse
from pathlib import Path
import os
from io import TextIOWrapper
import matplotlib.pyplot as plt
import numpy as np


def plot_graph(data_dict: dict, criteria: str, static_title: str, output_dir: Path, experiment_title: str = ""):
    lines_y = list(data_dict.values())
    lines_name = list(data_dict.keys())
    def get_max(x): return np.max(x)
    max_lines_y = [get_max(x) for x in lines_y]
    no_of_rounds = len(lines_y[0])
    assert all([len(y) == no_of_rounds for y in lines_y])
    x = [x for x in range(no_of_rounds)]

    cmap = plt.get_cmap('rainbow')
    cmap_list = np.linspace(0, 1, len(lines_y))
    for i in range(len(lines_y)):
        plt.plot(x, lines_y[i], c=cmap(cmap_list[i]),
                 label=lines_name[i], linewidth=1)
        plt.axhline(max_lines_y[i], c=cmap(cmap_list[i]),
                    linestyle="dashed", linewidth=0.75)

    plt.xlabel("Round")
    plt.ylabel("Accuracy/100")
    plt.title(f'{experiment_title} - {static_title}'.strip(' -:'))
    handles, labels = plt.gca().get_legend_handles_labels()
    handles, labels = zip(
        *sorted(zip(handles, labels), key=lambda t: t[1]))
    plt.legend(handles=handles, labels=labels, title=criteria)
    plt.legend(title=criteria)
    plt.savefig(f"{output_dir.joinpath(f'{experiment_title}-{criteria}.png')}")
    plt.show()


def extract_static(static_vars: list, file: TextIOWrapper):
    output_str = ""
    for var in static_vars:
        file.seek(0)
        val = None
        while True:
            line = file.readline()
            if (line == ""):
                break
            if (line.count(var) > 0):
                val = line.split(":")
                val = [item.strip() for item in val]
                val = val[-1]
                break
        if (val == None):
            raise ValueError(f'{var} not found in file')
        output_str += f'{var} : {val} ; '
    output_str = output_str.strip(" ; ")
    return output_str


def extract_criteria(criteria: str, file: TextIOWrapper):
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


def extract_server_test_acc(file: TextIOWrapper):
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


script_dir = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(
    description="Script for plotting the comparison graphs")
parser.add_argument("--data_dir", type=Path,
                    default=script_dir.joinpath("plot_data"))
parser.add_argument("--criteria", type=str, default="",
                    help="Criteria to compare on legend")
parser.add_argument("--static_vars", type=str, default="",
                    help="Static values to display between different lines")
parser.add_argument("--experiment_name", type=str, default="")
parser.add_argument("--output_dir", type=Path, default=script_dir)

if __name__ == "__main__":
    args = parser.parse_args()
    data_dir = args.data_dir
    criteria = args.criteria
    static_vars = args.static_vars
    experiment_name = args.experiment_name
    output_dir = args.output_dir

    file_list = os.listdir(data_dir)
    file_list[:] = [data_dir.joinpath(
        file) for file in file_list if (file.endswith('.out'))]
    if (len(file_list) < 1):
        raise IOError("No log files exist in the folder!")

    if (criteria == ""):
        print(f'Enter the criteria you want to compare : ')
        criteria = input()
    keys = []
    for file in file_list:
        with open(file, 'rt') as f:
            val = extract_criteria(criteria, f)
            keys.append(f'{val}')

    if (static_vars == ""):
        print(f'Enter list of static criteria (separated by comma): ')
        static_vars = input()
    static_title = ""
    if (static_vars != ""):
        static_vars = static_vars.split(",")
        static_vars[:] = [item.strip() for item in static_vars]
        with open(file_list[0], 'rt') as f:
            static_title = extract_static(static_vars, f)

    lines = {}
    for i in range(len(file_list)):
        with open(file_list[i], 'rt') as f:
            lines[keys[i]] = extract_server_test_acc(f)

    plot_graph(data_dict=lines, criteria=criteria,
               static_title=static_title, output_dir=output_dir, experiment_title=experiment_name)
