# digit recogniser
import pandas as pd
data = pd.read_csv("train.csv").as_matrix()
print(data)
data.shape
xtrain=data[:21000,1:]
xlable=data[:21000,0]
clf=DecisionTreeClassifier()
clf.fit(xtrain, xlable)
xtest=data[21000:,1:]
xlable=data[21000:,0]
d=xtest[8]
d.shape=(28,28)
from matplotlib import pyplot as plt
plt.imshow(255-d,cmap="gray")
print(clf.predict([xtest[8]]))
c=xtest[19560]
c.shape=(28,28)
plt.imshow(255-c,cmap="gray")
print(clf.predict([xtest[19560]]))
count=0
p=clf.predict(xtest)
for i in range(0,21000):
     if p[i]==xlable[i]:
         count = count + 1
else : 0
print("Accurecy:",count/21000*100)
import  os
os.getcwd()
from sklearn.tree import DecisionTreeClassifier

