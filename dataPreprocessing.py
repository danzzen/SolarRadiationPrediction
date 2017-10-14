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

traindata.drop(traindata[traindata['Pressure']<30.25].index , inplace=True)

traindata.drop(traindata[traindata['Speed']>20].index, inplace=True)

# traindata.drop(traindata[traindata['Time']>1200].index, inplace=True)
# traindata.drop(traindata[traindata['TimeSunRise']>1200].index, inplace=True)
# traindata.drop(traindata[traindata['TimeSunSet']>1200].index, inplace=True)
# sns.jointplot(x=traindata['Speed'],y=traindata['Radiation'])
# sns.plt.show()
##categorial variables ----->encoding VVIMP
## vvimp to thoda word of code bdane ko likh diya mene sklearn hi use karna h yha bhi
#convert categorial values into numeric data
def converter(var):
    from sklearn.preprocessing import LabelEncoder
    labelencoder=LabelEncoder()
    labelencoder.fit(traindata[var])
    print(list(labelencoder.classes_))
    traindata[var]=labelencoder.transform(traindata[var])
# print(traindata['Data'][0])
converter('Data')
sns.jointplot(x=traindata['Data'],y=traindata['Radiation'])
sns.plt.show()
traindata.drop(traindata[traindata['Data']>1200].index, inplace=True)
converter('Time')
converter('TimeSunRise')
converter('TimeSunSet')
sns.jointplot(x=traindata['Time'],y=traindata['Radiation'])
sns.jointplot(x=traindata['TimeSunRise'],y=traindata['Radiation'])
sns.jointplot(x=traindata['TimeSunSet'],y=traindata['Radiation'])
traindata.drop(traindata[traindata['Radiation']>1200 | (traindata['TimeSunSet']>40 & traindata['Radiation']>1000)].index,inplace=True)
sns.plt.show()
