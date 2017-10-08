# Yes! lets do it
#path gyi

###########import as many files as you can
print ("checking for numpy")
try:
    import numpy as np
except ImportError:
    print ("you should install numpy before continuing")

print ("checking for scipy")
try:
    import scipy
except:
    print ("you should install scipy before continuing")

print ("checking for sklearn")
try:
    import sklearn
except:
    print ("you should install sklearn before continuing")
try:
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = (10.0, 8.0)
except:
    print("you shoild install matplotlib before continuing")
try:
    import pickle
except:
    print("you should install pickle before continuing")
try:
    import pandas as pd
except:
    print("you should install pylear2 before continuing")
try:
    import seaborn as sns
except:
    print("bsdk bhut bol liya phle saare install kar")

####sk.learn kahne ko coder project karta h but bc sara kam to ye library kati h
import sklearn
from sklearn.model_selection import train_test_split
"""chlo bc sab library import kar di mughe itna hi aata h
########### lets have a look at data=====>>>>"""

####loading data
df = pd.read_csv("SolarPrediction.csv")
"""print(train.head())"""

"""split data into training and testing data set us coloumn se split karna jo label ho"""


traindata, testdata = train_test_split(df, test_size=0.3)

print ('The train data has {0} rows and {1} columns'.format(traindata.shape[0],traindata.shape[1]))
print ('----------------------------')
print ('The test data has {0} rows and {1} columns'.format(testdata.shape[0],testdata.shape[1]))

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
sns.plt.show() ### ye graph dikhayega label ke distribution ka nhi samgh aaye to rat lena
# now we have to normalize it so it looks like normaly distributed. PK chutiya
target=np.log(traindata['Radiation'])
sns.distplot(target)
sns.plt.show()
print(target.skew())

### we can see and get a good idea about the diatribution of value of radiation over different features. Ghanta graph dekhkar jo aata tha vo bhi bhul gya

