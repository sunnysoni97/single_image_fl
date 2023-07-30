import argparse
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
from multiseed_helper import combine_seeds


def plot_graph(data_dict: dict, legend_title: str, output_dir: Path, experiment_title: str = ""):
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
    plt.title(f'{experiment_title}')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles, labels = zip(
        *sorted(zip(handles, labels), key=lambda t: t[1]))
    plt.legend(handles=handles, labels=labels, title=legend_title)
    plt.savefig(
        f"{output_dir.joinpath(f'{experiment_title}-{legend_title}.png')}")
    plt.show()

    text_file_path = output_dir.joinpath(
        f'{experiment_title}-{legend_title}.txt')

    with open(text_file_path, 'wt') as f:
        f.write(f'Acc. Stats for the experiment : {experiment_title}\n\n')
        for line_name, line_values, max_value in zip(lines_name, lines_y, max_lines_y):
            f.write(
                f'Line name : {line_name}\nAcc values : {line_values}\nMax Acc. : {max_value} \nMax Round : {np.where(line_values == max_value)[0][0]}\n\n')


script_dir = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(
    description="Script for plotting the comparison graphs")
parser.add_argument("--data_dir", type=Path,
                    default=script_dir.joinpath("plot_data"))
parser.add_argument("--output_dir", type=Path, default=script_dir)

parser.add_argument("--criteria", type=str, default="",
                    help="Criteria to compare on legend")
parser.add_argument("--criteria_labels", type=str, default="")
parser.add_argument("--seed_count", type=int, default=None)
parser.add_argument("--experiment_name", type=str,
                    default="Accuracy Comparison")

if __name__ == "__main__":
    args = parser.parse_args()
    data_dir = args.data_dir
    criteria = args.criteria
    criteria_labels = args.criteria_labels
    seed_count = args.seed_count
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

    if ',' in criteria:
        criteria = criteria.split(sep=',')
        criteria = [criteria[i].strip() for i in range(len(criteria))]

        if (criteria_labels == ""):
            print(f'Enter labels for each configuration (optional) : ')
            criteria_labels = input()

        if (criteria_labels != ""):
            criteria_labels = criteria_labels.split(sep=',')
            criteria_labels = [criteria_labels[i].strip()
                               for i in range(len(criteria_labels))]
        else:
            criteria_labels = []

    if (seed_count == None):
        print(f'Enter number of seeds for each experiment : ')
        seed_count = int(input())

    if (criteria_labels):
        legend_title = "Setup"
    else:
        if (type(criteria) == type([])):
            legend_title = '-'.join(criteria)
        else:
            legend_title = criteria

    lines = combine_seeds(n_seeds=seed_count, dir_path=data_dir,
                          criteria=criteria, criteria_labels=criteria_labels)

    plot_graph(data_dict=lines, legend_title=legend_title,
               output_dir=output_dir, experiment_title=experiment_name)
