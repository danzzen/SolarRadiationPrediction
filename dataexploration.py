## file ka naam sunke hi path gyi code ghanta likh paunga
from main import traindata
import sys
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
sns.distplot(traindata['Radiation'])
# sns.plt.show() ### ye graph dikhayega label ke distribution ka nhi samgh aaye to rat lena
# now we have to normalize it so it looks like normaly distributed. PK chutiya
target=np.log(traindata['Radiation'])
sns.distplot(target)
# sns.plt.show()
print(target.skew())

### we can see and get a good idea about the diatribution of value of radiation over different features. Ghanta graph dekhkar jo aata tha vo bhi bhul gya

"""
2. 
    i) seperate numerical and categorial data
   ii) plot correlation matrix   
"""

numeric_data= traindata.select_dtypes(include=[np.number])
cat_data= traindata.select_dtypes(exclude=[np.number])
print("numeric value coloums = ",numeric_data.shape[1])
print("cat value coloums = ",cat_data.shape[1])


#correlation plot
corr=numeric_data.corr()
sns.heatmap(corr)
# sns.plt.show()

print (corr['Radiation'].sort_values(ascending=False)[:7], '\n') #top 15 values
"""negative ko chod do postive will impact the """
