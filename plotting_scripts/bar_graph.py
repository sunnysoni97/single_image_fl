import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("poster")
 
# creating the dataset

# data = {'Single Image':74.8,'500 CIFAR100 Samples':73.2}
# data = {'Single Image':73.2,'500 CIFAR100 Samples':71.3}
data = {'Single Image':68.6,'500 CIFAR100 Samples':66.5}
courses = list(data.keys())
values = list(data.values())
  
fig = plt.figure(figsize=(8,6))
fig.subplots_adjust(left=0.16)

# creating the bar plot
plt.bar(courses, values, color =['lightgreen','lightblue'], align='center', width=0.25)
 
plt.ylim((65,75))
plt.xlim(-0.60,1.60)
plt.ylabel("Evaluation Accuracy (in %)", fontsize=18, labelpad=18)
plt.yticks(fontsize=16)
plt.xticks(fontsize=18)
plt.savefig("bar_graph.pdf", bbox_inches="tight", dpi=300)
plt.show()

