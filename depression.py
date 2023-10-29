import pandas as pd
import numpy as np

def variance(data):
	n = len(data)
	mean =  sum(data)/n
	deviations = [(x-mean)**2 for x in data]
	variance = sum(deviations)/n
	return variance



df = pd.read_csv('res.csv')
ref = np.ones(len(df['1-2'].to_numpy()))
length = len(ref)

a1 = sum(ref - df['1-2'].to_numpy())/length
a2 = sum(ref - df['1-3'].to_numpy())/length
a3 = sum(ref - df['1-4'].to_numpy())/length
a4 = sum(ref - df['1-5'].to_numpy())/length



# for index, row in df.iterrows():
#     a1.append(row['1-2'])
# 	# a2.append(row['1-3'])

#print('variance 1-2 = ', variance(a1))
#print('variance 1-3 = ', variance(a2))
print('Среднее отклонение при 5 ps:', a1)
print('Среднее отклонение при 10 ps:', a2)
print('Среднее отклонение при 15 ps:', a3)
print('Среднее отклонение при 20 ps:', a4)
