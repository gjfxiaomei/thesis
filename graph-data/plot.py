import matplotlib.pyplot as plt
# import seaborn as sns
import pandas


for da in ['Hangzhou', 'Jinan', '6x6', '6x6-b']:
    gcn = pandas.read_csv(f"./{da}/only-gcn-new.csv", index_col=None)
    ind = pandas.read_csv(f"./{da}/only-ind-new.csv", index_col=None)
    col = pandas.read_csv(f"./{da}/only-col-new.csv", index_col=None)
    gat = pandas.read_csv(f"./{da}/GAT-Road.csv", index_col=None)
    plt.plot(gcn.x, gcn.y, ls='-', label='GCN', color='purple')
    plt.plot(ind.x, ind.y, ls='-', label='Individual RL', color='green')
    plt.plot(col.x, col.y, ls='-', label='Colight', color='blue')
    plt.plot(gat.x, gat.y, ls='-', label='GAT-Road', color='red')
    plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('Travel Time')
    plt.savefig(f'./{da}/conv-{da}.pdf', dpi=800, format='pdf')
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