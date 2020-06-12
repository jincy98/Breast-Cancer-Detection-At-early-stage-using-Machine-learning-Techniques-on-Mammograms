import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

df = pd.read_csv('cancer1_data.csv')
print df.head(7)
print df.shape
print df['label'].value_counts()
sns.set (style="whitegrid",color_codes=True)
sns.countplot(df['label'], label='count')
plt.title('label vs count')
plt.savefig('graph1 label&count.png')

LabelEncoder_y = LabelEncoder()
df.iloc[:,1] = LabelEncoder_y.fit_transform(df.iloc[:,1].values)

print df.iloc[:,1]

sns.set (style="whitegrid",color_codes=True)
sns.pairplot(df.iloc[:,1:6], hue='label')
plt.title('relation')
plt.savefig('relationship')

#dataset splitting  x indepedent and y dependent datasets.
x = df.iloc[:,2:7].values
y = df.iloc[:,1].values

#split the dataset into 75% training and 25% testing.
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25 ,random_state = 0)

#scale the data(feature scaling)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
print x_train

#create a function for the models.
def models(x_train,y_train):
	#logistic regression
	log = LogisticRegression(random_state=0)
	log.fit(x_train,y_train)

	#decision tree
	tree = DecisionTreeClassifier(criterion = 'entropy',random_state = 0)
	tree.fit(x_train,y_train)

	#random forest 
	forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
	forest.fit(x_train,y_train)

	#print the models accuracies on the training data
	print('1:  logistic regression training accuracy:',log.score(x_train,y_train))
	print('2:  decision tree training accuracy:',tree.score(x_train,y_train))
	print('3: random forest training accuracy:',forest.score(x_train,y_train))
	return log,tree,forest
	#getting all of the models.
model = models(x_train,y_train)

#test model accuracy on test data on confusion matrix
for i in range(len(model)):
	print('model',i)
	cm = confusion_matrix(y_test, model[i].predict(x_test))
	TP = cm[0][0]
	TN = cm[1][1]
	FP = cm[0][1]
	FN = cm[1][0]
	first = TP + TN
	#print(first)
	second= TP+TN+FP+FN
	#print(second)
	ac = (float(first)/float(second))
	print(cm)
	print('testing accuracy is :', ac)
	print()
	
#show another way to get the metrix of the models
for i in range(len(model)):
	print('model',i)
	print(classification_report(y_test,model[i].predict(x_test)))
	print(accuracy_score(y_test,model[i].predict(x_test)))
	print()

#print the prediction of random forest classifier model
pred = model[2].predict(x_test)
print('prediction of Breast cancer',pred)
print()
print('actual data set',y_test)