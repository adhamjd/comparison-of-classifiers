from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from mlxtend.classifier import StackingCVClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
import pandas as pd
#imported libraries to be used






data = pd.read_csv("spam.csv")
print(data)
#assigning each classifier to a variable
clf = KNeighborsClassifier(n_neighbors=5)
clf1 = DecisionTreeClassifier()
clf2 = RandomForestClassifier(max_depth= 2,random_state=0)
clf4 = MLPClassifier(hidden_layer_sizes=(56,), activation='relu', solver='adam', max_iter=500)
#assigning the different classifiers to the stacking classifier
sclf = StackingCVClassifier(classifiers=[clf, clf1, clf2,], use_probas=True, meta_classifier=LogisticRegression())


#deciding the values for both X and Y depedning on the dataset.
X = data.values[:, :56]
Y = data.values[:,57]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
clf = clf.fit(X_train, Y_train)
Y_prediction = clf.predict(X_test)
print("Train/test accuracy:",accuracy_score(Y_test,Y_prediction))


from sklearn.model_selection import ShuffleSplit          #the import of shuffle split
cv = ShuffleSplit(n_splits=5, test_size=0.2)              #shufflesplit is the equivilant of picking (in this case 5 splits) randomly




from sklearn.model_selection import cross_val_score
maLP = cross_val_score(clf4, X, Y, cv=cv)
maLPrecision = cross_val_score(clf4, X, Y, cv=cv, scoring='precision_macro')
scores = cross_val_score(clf, X, Y, cv=cv)
scoresprecision = cross_val_score(clf, X, Y, cv=cv, scoring='precision_macro')
scoresdecisiontree = cross_val_score(clf1, X, Y, cv=cv)
scoresdecisiontreeprecision = cross_val_score(clf1, X, Y, cv=cv, scoring='precision_macro')
Randomfscores = cross_val_score(clf2, X, Y, cv=cv,)
Randomfscoresprecision = cross_val_score(clf2, X, Y, cv=cv, scoring='precision_macro')

for clf, label in zip([sclf], ['StackingClassifier']):
    StackingScores = cross_val_score(sclf, X, Y, cv=cv, scoring='accuracy')
    StackingPrecisionScores = cross_val_score(sclf, X, Y, cv=cv, scoring='precision_macro')

    print("[%s] Stacking CFV accuracy mean: %0.3f (+/- %0.f)" % (label, StackingScores.mean(), StackingScores.std()))
    print("[%s] Stacking CFV Precision mean: %0.3f (+/- %0.3f)" % (
    label, StackingPrecisionScores.mean(), StackingPrecisionScores.std()))

print()
print("MLP accuracy mean:",maLP.mean())
print("MLP Precison mean:",maLPrecision.mean())
print("Random Forest accuracy mean:",Randomfscores.mean())
print("Random Forest Precision mean:",Randomfscoresprecision.mean())
print("Decision Tree accuracy mean:",scoresdecisiontree.mean())
print("Decision Tree Precision mean:",scoresdecisiontreeprecision.mean())
print("Cross fold validation accuracy mean:",scores.mean())
print("Cross fold validation  precision mean",scoresprecision.mean())
