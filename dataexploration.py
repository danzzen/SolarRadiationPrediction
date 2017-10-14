## file ka naam sunke hi path gyi code ghanta likh paunga
from main import traindata
from anova import anova
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 8.0)
import seaborn as sns
import numpy as np

###check if any null value dataset have or not.     Mughe ye ache se nhi aata but kisi tarh pel deta hu stackoverflow apna bhai h
print(traindata.columns[traindata.isnull().any()])

#lets check percentage of missing value in coloumns V.imp bc vese ki karte h koi jarurat nhi h
noOfmiss=traindata.isnull().sum()/len(traindata)
noOfmiss=noOfmiss[noOfmiss > 0]
print(noOfmiss)
"""visualize missing values. ghanta bas dekh lo bad me to library hi use karna h"""
#plot the missing value count
if(len(noOfmiss)>0):
    noOfmiss=noOfmiss.to_frame()
    noOfmiss.columns=['count']
    noOfmiss.index.names=['Name']
    noOfmiss['Name']=noOfmiss.index
    sns.set(style="whitegrid", color_codes=True)
    sns.barplot(x = 'Name', y = 'count', data=noOfmiss)
    plt.xticks(rotation = 90)
    sns.plt.show()

#plot the label now
# sns.distplot(traindata['Radiation'])
# sns.plt.show() ### ye graph dikhayega label ke distribution ka nhi samgh aaye to rat lena
# now we have to normalize it so it looks like normaly distributed. PK chutiya
target=np.log(traindata['Radiation'])
# sns.distplot(target)
# sns.plt.show()
print(target.skew())

### we can see and get a good idea about the diatribution of value of radiation over different features. Ghanta graph dekhkar jo aata tha vo bhi bhul gya

"""
2. 
    i) seperate numerical and categorial data
   ii) plot correlation matrix   
"""

numeric_data = traindata.select_dtypes(include=[np.number])
cat_data = traindata.select_dtypes(exclude=[np.number])
print("numeric value coloums = ", numeric_data.shape[1])
print("cat value coloums = ", cat_data.shape[1])

"""NUMERIC DATA EXPLORATION"""
#correlation plot
corr=numeric_data.corr()
sns.heatmap(corr)
sns.plt.show()

print (corr['Radiation'].sort_values(ascending=False), '\n') #top 15 values
"""negative ko chod do postive will impact the """
##highest correlated is Temperature 0.734
print ((traindata['Temperature'].unique())) #range is 34 to 71, vese hi likha light le

#lets chek the mean Radiation per temperature and plot it
pivot = traindata.pivot_table(index='Temperature',values='Radiation',aggfunc=np.median)
print(pivot)

# pivot.plot(kind='bar',color='blue')

###issi tarf saare positive correlationscore valo ko lekar graph draw kra. Vese hi timepass na ho rha ho to kar lena

"""
    1 dot vale graph for distribution of each feature

"""
sns.jointplot(x=traindata['Pressure'],y=traindata['Radiation'])
sns.plt.show()
##for each correalted variables this can be seen, Bas ek bar dekh lo smgh vese bhi kuch nhi aana

""" CATEGORIAL DATA EXPLORATION"""
print(cat_data.describe())


rad_pivot = traindata.pivot_table(index='Data',values='Radiation',aggfunc=np.median)
rad_pivot.plot(kind='bar', color='red')
sns.plt.show()
cat = [f for f in traindata.columns if traindata.dtypes[f] == 'object']

cat_data['Radiation'] = traindata['Radiation'].astype(float)
k = anova(cat_data,cat)
k['disparity'] = np.log(1./k['pval'].values)
sns.barplot(data=k, x='features', y='disparity')
plt.xticks(rotation=90)
plt.show()
