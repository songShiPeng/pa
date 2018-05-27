fileDir = '/Users/songshipeng/Downloads/tt/'
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv(fileDir+'train.csv')
print(train_data.info())
train_data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar()
plt.show()
train_data[['Age','Survived']].groupby(['Age']).mean().plot.bar()

plt.show()