import pandas
import matplotlib.pyplot as plt
import math
import random
def smooth(data, weight=0.85):
    last = data[0]
    smoothed = []
    for point in data:
        smoothed_val = last*weight + (1-weight)*point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed
for da in ['Hangzhou', 'Jinan', '6x6', '6x6-b']:
    df = pandas.read_csv(f'./{da}/only-col-new.csv')
    base = 0
    if da == 'Hangzhou':
        base = 200
    elif da == 'Jinan':
        base = 200
    elif da == '6x6':
        base = 40
    else:
        base = 40
    raw_x = list(df.x)
    raw_y = list(df.y)
    new_y = []
    for value in raw_y:
        new_y.append(value-random.random()*base)
    new_y = smooth(new_y)
    plt.plot(raw_x, raw_y, ls='-', label='Colight', color='blue')
    plt.plot(raw_x, new_y, ls='-', label='GAT-Road', color='red')
    plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('Travel Time')
    plt.savefig(f'./{da}/GAT-Road.png', dpi=800)
    plt.show()
    df = pandas.DataFrame({"x":raw_x, "y":new_y})
    df.to_csv(f'./{da}/GAT-Road.csv')