import pandas as pd
from sklearn import preprocessing, model_selection, neighbors
import numpy as np

df = pd.read_csv('')
X = np.array(df.drop(['class'],1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.3)

clf = neighbors.KNeighborsClassifier()

clf.fit(X_train,y_train)

accuracy = clf.score(X_test,y_test)

print(accuracy)

# example_measures = np.array()
# example_measures  = example_measures.reshape(1,-1)

prediction = clf.predict(example_measures)

print(prediction)
