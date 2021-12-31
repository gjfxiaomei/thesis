import pandas
import math

def smooth(data, weight=0.9):
    last = data[0]
    smoothed = []
    for point in data:
        smoothed_val = last*weight + (1-weight)*point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed
for da in ['Hangzhou', 'Jinan', '6x6', '6x6-b']:
    for method in ['ind', 'gcn', 'col']:
        df = pandas.read_csv(f'./{da}/only-{method}.csv')
        raw_x = list(df.x)
        raw_y = list(df.y)
        new_x = []
        final_x = []
        final_y = []
        for item in raw_x:
            r, i = math.modf(item)
            new_x.append(int(i))
        print(new_x)

        for (i, value) in enumerate(new_x):
            if value not in final_x:
                final_x.append(value)
                final_y.append(raw_y[i])
            else:
                final_y[-1] = (raw_y[i] + final_y[-1])/2


        final_y = smooth(final_y, weight=0.8)

        final_data = {"x":final_x, "y":final_y}
        df = pandas.DataFrame(final_data)
        df.to_csv(f'./{da}/only-{method}-new.csv')