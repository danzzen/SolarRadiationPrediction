# like this part but ye bhi nhi aata bas naam sunke acha lgta h #padhupart#bhuttough#nhihopayegatumse
"""
Removeing outliers
Creating new features
Filling missing values
"""
from main import traindata,pd,np,sns
#our data dont have any missing values so no need to filling values
"""1 outiler removal"""
traindata.drop(traindata[traindata['Radiation']>1400].index, inplace=True)

traindata.drop(traindata[traindata['Pressure']<30.25].index, inplace=True)

traindata.drop(traindata[traindata['Speed']>20].index, inplace=True)

##categorial variables ----->encoding VVIMP
## vvimp to thoda word of code bdane ko likh diya mene sklearn hi use karna h yha bhi
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
def factorize(data, var, fill_na = None):
    if fill_na is not None:
        data[var].fillna(fill_na,inplace=True)
    labelencoder.fit(data[var])
    data[var] = labelencoder.transform(data[var])
    return data
for f in traindata.columns:
    if traindata.dtypes[f]=='object':
        traindata=factorize(traindata,f)
print(traindata.head())
