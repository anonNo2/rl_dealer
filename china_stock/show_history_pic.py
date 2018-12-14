
import pandas as pd
import matplotlib.pyplot as plt
filename = "000002.SZ.csv"

data = pd.read_csv(filename)
data['trade_date'] = pd.to_datetime(data['trade_date'],format='%Y%m%d')
data.set_index('trade_date')
# print(data.iloc[0:50])
data = data.sort_index()
# print(data.iloc[0:50])
date = data.iloc[:]["trade_date"]
closed = data.iloc[:]["close"]
#id = [i for i in range(len(closed))]

plt.plot(date,closed,'b-')
plt.show()

