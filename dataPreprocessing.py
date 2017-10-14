# like this part but ye bhi nhi aata bas naam sunke acha lgta h #padhupart#bhuttough#nhihopayegatumse
"""
Removeing outliers
Creating new features
Filling missing values
"""
from main import traindata,pd,np,sns,testdata
#our data dont have any missing values so no need to filling values
"""1 outiler removal"""
alldata2 = traindata.append(testdata)
alldata2.drop(alldata2[alldata2['Radiation']>1400].index, inplace=True)

alldata2.drop(alldata2[alldata2['Pressure']<30.25].index , inplace=True)

alldata2.drop(alldata2[alldata2['Speed']>20].index, inplace=True)

# alldata2.drop(alldata2[alldata2['Time']>1200].index, inplace=True)
# alldata2.drop(alldata2[alldata2['TimeSunRise']>1200].index, inplace=True)
# alldata2.drop(alldata2[alldata2['TimeSunSet']>1200].index, inplace=True)
# sns.jointplot(x=alldata2['Speed'],y=alldata2['Radiation'])
# sns.plt.show()
##categorial variables ----->encoding VVIMP
## vvimp to thoda word of code bdane ko likh diya mene sklearn hi use karna h yha bhi
#convert categorial values into numeric data
def converter(var):
    from sklearn.preprocessing import LabelEncoder
    labelencoder=LabelEncoder()
    labelencoder.fit(alldata2[var])
    print(list(labelencoder.classes_))
    alldata2[var]=labelencoder.transform(alldata2[var])
# print(alldata2['Data'][0])
converter('Data')
converter('Time')
converter('TimeSunRise')
converter('TimeSunSet')
# sns.jointplot(x=alldata2['Data'],y=alldata2['Radiation'])
# sns.plt.show()

# sns.jointplot(x=alldata2['Data'],y=alldata2['Radiation'])

# sns.jointplot(x=alldata2['Time'],y=alldata2['Radiation'])
# sns.jointplot(x=alldata2['TimeSunRise'],y=alldata2['Radiation'])
# sns.jointplot(x=alldata2['TimeSunSet'],y=alldata2['Radiation'])
alldata2.drop(alldata2[alldata2['Radiation']>1200].index, inplace=True)
alldata2.drop(alldata2.loc[(alldata2['Radiation']>int(1200.0)) | ((alldata2['TimeSunRise']>int(40)) & (alldata2['Radiation']>int(1000)))].index,inplace=True)
# sns.jointplot(x=alldata2['TimeSunRise'],y=alldata2['Radiation'])
alldata2.drop(alldata2.loc[alldata2['Radiation']>1200].index, inplace=True)
sns.jointplot(x=alldata2['TimeSunSet'],y=alldata2['Radiation'])
sns.plt.show()
from sklearn.model_selection import train_test_split

traindata1, testdata1 = train_test_split(alldata2, test_size=0.3)
#drop unwanted coloumns
drop_cols=["UNIXTime","Humidity","WindDirection(Degrees)","Time"]

traindata1.drop(drop_cols,axis=1,inplace=True)
testdata1.drop(drop_cols,axis=1,inplace=True)
#get numeric features
numeric_features = [f for f in traindata1.columns if traindata1[f].dtype != object]

#transform the numeric features using log(x + 1)
from scipy.stats import skew
skewed = traindata1[numeric_features].apply(lambda x: skew(x.dropna().astype(float)))
skewed = skewed[skewed > 0.75]
skewed = skewed.index
traindata1[skewed] = np.log1p(traindata1[skewed])
testdata1[skewed] = np.log1p(testdata1[skewed])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(traindata1[numeric_features])
scaled = scaler.transform(traindata1[numeric_features])

for i, col in enumerate(numeric_features):
       traindata1[col] = scaled[:,i]

numeric_features.remove('Radiation')
scaled = scaler.fit_transform(testdata1[numeric_features])

for i, col in enumerate(numeric_features):
      testdata1[col] = scaled[:,i]
