import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_context("poster")

# creating the dataset

# data = {'Single Image':74.8,'500 CIFAR100 Samples':73.2} 100
# data = {'Single Image':73.2,'500 CIFAR100 Samples':71.3} 50
# data = {'Single Image':68.6,'500 CIFAR100 Samples':66.5} 20

data_dict = {
    "Source": [
        "Single Image",
        "CIFAR100 Samples",
        "Single Image",
        "CIFAR100 Samples",
        "Single Image",
        "CIFAR100 Samples",
    ],
    "Accuracy": [74.8, 73.2, 73.2, 71.3, 68.6, 66.5],
    "InitPercent": [100, 100, 50, 50, 20, 20],
}

data_df = pd.DataFrame(data=data_dict)


fig = plt.figure(figsize=(10, 7.5))
fig.subplots_adjust(left=0.16, bottom=0.16)

g = sns.barplot(
    data=data_df,
    x="InitPercent",
    y="Accuracy",
    hue="Source",
    width=0.4,
    palette=["lightgreen", "lightblue"],
    saturation=1.5,
)

g.bar_label(g.containers[0], fontsize=13)
g.bar_label(g.containers[1], fontsize=13)

plt.ylim((64.1, 75.9))
plt.xlim(-0.30, 2.30)
plt.ylabel("Evaluation Accuracy (in %)", fontsize=18, labelpad=18)
plt.xlabel("FedAvg Initialisation Rate (in %)", fontsize=18, labelpad=18)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.legend(
    loc="upper left", fontsize=14, title="Distillation Dataset", title_fontsize=12
)
plt.savefig("bar_graph.pdf", bbox_inches="tight", dpi=300)
plt.show()
