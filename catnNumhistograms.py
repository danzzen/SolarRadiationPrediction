#lets get started bro , jada josh me mat aa
from main import traindata
from main import pd
from main import sns
from main import plt
#create numeric plots , bah mat bsdk

num=[f for f in traindata.columns if traindata.dtypes[f]!='object']
nd=pd.melt(traindata,value_vars=num)
n1 = sns.FacetGrid (nd, col='variable', col_wrap=4, sharex=False, sharey = False)
n1 = n1.map(sns.distplot, 'value')
sns.plt.show()

#create box plots for categorials values
def boxplot(x,y,**kwargs):
            sns.boxplot(x=x,y=y)
            x = plt.xticks(rotation=90)

cat = [f for f in traindata.columns if traindata.dtypes[f] == 'object']

p = pd.melt(traindata, id_vars='Radiation', value_vars=cat)
g = sns.FacetGrid (p, col='variable', col_wrap=2, sharex=False, sharey=False, size=5)
g = g.map(boxplot, 'value','Radiation')
# g.plt.show()
