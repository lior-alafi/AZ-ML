import pandas as pd
import numpy as np

from apyori import apriori


df = pd.read_csv('../data/Market_Basket_Optimisation.csv',header=None)
print(df.head())

transactions = []
for row in range(0,df.shape[0]):
    transactions.append([str(df.iloc[row,col]) for col in range(0,df.shape[1])])
# 3 failed per day * 7 / len(transactions)
MIN_SELECTION = 3
DAYS_OF_WEEK = 7

rules = apriori(transactions=transactions,min_support=(MIN_SELECTION*DAYS_OF_WEEK)/df.shape[0],min_confidence=0.2,min_lift=3.0,min_length=2,max_length=2)



## Putting the results well organised into a Pandas DataFrame
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
resultsinDataFrame = pd.DataFrame(inspect(list(rules)), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

resultsinDataFrame.nlargest(n = 10, columns = 'Lift')
print(resultsinDataFrame.sort_values(by=['Lift'],ascending=False))