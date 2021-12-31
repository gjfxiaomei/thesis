import matplotlib.pyplot as plt
# import seaborn as sns
import pandas

def smooth(data, weight=0.7):
    last = data[0]
    smoothed = []
    for point in data:
        smoothed_val = last*weight + (1-weight)*point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

w = pandas.read_csv("./with.csv", index_col=None)
we = pandas.read_csv("./without.csv", index_col=None)

plt.plot(w.Step, smooth(w.Value), ls='-', label='with Edge Weight', color='green')
plt.plot(we.Step, smooth(we.Value), ls='-', label='without Edge Weight', color='red')


plt.legend()
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.savefig('./diff.pdf', dpi=800, format='pdf')
plt.show()

# gcn['method'] = "GCN"
# ind['method'] = "Individual"
# col['method'] = "Colight"
# total_data = pandas.concat([gcn, ind, col])
# # print(total_data)
# fig = plt.figure()
# ax = sns.boxplot(x='x',y='y',hue='method',fliersize=5.0, data=total_data)
# font = {
# 'weight' : 'normal',
# 'size'   : 15,
# }
# plt.legend(title=None)
# plt.show()
# fig.savefig('./plot.pdf', dpi=800, format='pdf')