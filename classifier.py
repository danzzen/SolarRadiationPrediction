# ye sab kar lete h sklearn use karke
from dataPreprocessing import traindata1,testdata1
from main import pd,np
# create a label frame
label_df=pd.DataFrame(index=traindata1.index,columns=['Radiation'])
label_df['Radiation']=traindata1['Radiation']
print("Training set size:", traindata1.shape)
print("Test set size:", testdata1.shape)
from sklearn.linear_model import Lasso
traindata1.drop("Radiation",axis=1,inplace=True)
print(traindata1.head())
print(label_df.head())
#found this best alpha through cross-validation
best_alpha = 0.00099

regr = Lasso(alpha=best_alpha, max_iter=50000)
regr.fit(traindata1, label_df)

from sklearn.metrics import mean_squared_error
def rmse(y_test,y_pred):
      return np.sqrt(mean_squared_error(y_test,y_pred))
y_pred = regr.predict(traindata1)
y_test = label_df
# from sklearn.metrics import accuracy_score
# print(accuracy_score(y_test, y_pred))
#

print("Lasso score on training set: ", rmse(y_test, y_pred))
y_test=testdata1['Radiation']
testdata1.drop("Radiation",axis=1,inplace=True)
y_pred=regr.predict(testdata1)

print("Lasso score on training set: ", rmse(y_test, y_pred))

