
import pandas as p
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
ds_t=p.read_csv(r"C:\Users\srava\OneDrive\Desktop\ML\Dataset\train.csv")
ds_te=p.read_csv(r"C:\Users\srava\OneDrive\Desktop\ML\Dataset\test.csv")
#Replacing Sex and Embarked with numerical values
ds_t['Sex'] = ds_t['Sex'].replace(["female", "male"], [0, 1])
ds_t['Embarked'] = ds_t['Embarked'].replace(['S', 'C', 'Q'], [1, 2, 3])
matrix = ds_t.corr()
print(matrix)
#Heatmap showing the correlation between variables
sns.heatmap(matrix, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag')
plt.show()
# We need to keep the "Sex" feature because it has high correlation with survived which is the feature which is to be found
sns.histplot(data=ds_t, x="Survived", hue="Sex")
plt.show()
classifier=GaussianNB()
ds_t.dropna(axis=0,inplace=True)
ds_te['Sex'] = ds_t['Sex'].replace(["female", "male"], [0, 1])
ds_te['Embarked'] = ds_t['Embarked'].replace(['S', 'C', 'Q'], [1, 2, 3])
ds_te.dropna(axis=0,inplace=True)
x=ds_t.loc[:,['Age', 'Embarked', 'Fare', 'Parch', 'Sex', 'SibSp']]
y=ds_t['Survived']
x_test=ds_te.loc[:,['Age', 'Embarked', 'Fare', 'Parch', 'Sex', 'SibSp']]
y_test=ds_te
classifier.fit(x,y)
y_pred=classifier.predict(x_test)
print('accuracy is',accuracy_score(y[:13], y_pred))