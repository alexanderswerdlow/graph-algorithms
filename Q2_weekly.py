from numpy import array, mean, sqrt, log
import datetime
from pathlib import Path
import csv

stocks = {}
pathlist = Path("finance_data/data").glob('*.csv')
for path in sorted(pathlist):
    with open(str(path)) as f:
        reader = csv.reader(f)
        last_close = None
        time_vals = []
        has_started = False
        num_added = 0
        for idx, row in enumerate(reader):

            if idx == 0:
                continue

            try:
                cur_date = datetime.datetime.strptime(row[0], '%Y-%m-%d')
            except:
                cur_date = datetime.datetime.strptime(row[0], '%m/%d/%y')

            if cur_date.weekday() == 0:
                if not has_started:
                    last_close = float(row[4])
                    has_started = True
                else:
                    num_added += 1
                    cur_close = float(row[4])
                    cur_date = None

                    time_vals.append(log(1+(cur_close - last_close)/last_close))
                    last_close = cur_close

        if num_added == 142:
            stocks[path.stem] = time_vals

file_names = list(stocks.keys())
with open("weights_weekly.txt", "w") as file:
    for i in range(len(file_names)):
        for j in range(i + 1, len(file_names)):
            r_i, r_j = array(stocks[file_names[i]]), array(stocks[file_names[j]])
            ro_i_j = (mean(r_i * r_j) - mean(r_i) * mean(r_j)) / sqrt((mean(r_i**2) - mean(r_i)**2) * (mean(r_j**2) - pow(mean(r_j), 2)))
            w_i_j = sqrt(2*(1-ro_i_j))
            if w_i_j:
                file.write(f'{file_names[i]} {file_names[j]} {w_i_j}\n')
