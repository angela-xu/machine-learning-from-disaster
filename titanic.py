import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

train = 'train.csv' 
train = pd.read_csv(train)
test = 'test.csv'
test = pd.read_csv(test)
print(train.head())
print(test.head())

print(train['Survived'].value_counts())
print(train['Survived'].value_counts(normalize=True))

print(train['Survived'][train['Sex'] == 'male'].value_counts())
print(train['Survived'][train['Sex'] == 'female'].value_counts())
print(train['Survived'][train['Sex'] == 'male'].value_counts(normalize=True))
print(train['Survived'][train['Sex'] == 'female'].value_counts(normalize=True))

train["Child"] = float('NaN')
train['Child'][train['Age'] < 18] = 1    # Assign 1 to passengers under 18, 0 to those 18 or older.
train['Child'][train['Age'] >= 18] = 0
print(train['Child'])
print(train["Survived"][train["Child"] == 1].value_counts(normalize = True))
print(train['Survived'][train['Child'] == 0].value_counts(normalize = True))

# Gender classification
test_one = test
test_one['Survived'] = 0
test_one['Survived'][test_one['Sex'] == 'female'] = 1
print(test_one['Survived'])

# Decision tree classification
train["Age"] = train["Age"].fillna(train["Age"].median())    # Impute missing values of Age with median
train["Sex"][train["Sex"] == "male"] = 0
train['Sex'][train['Sex'] == 'female'] = 1

train["Embarked"] = train['Embarked'].fillna('S')    # Impute the Embarked variable
train["Embarked"][train["Embarked"] == "S"] = 0    # Convert the Embarked classes to integer form
train['Embarked'][train['Embarked'] == 'C'] = 1
train['Embarked'][train['Embarked'] == 'Q'] = 2

print(train['Sex'])
print(train['Embarked'])
print(train)

target = train['Survived'].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

# First decision tree
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)
print(my_tree_one.feature_importances_)    # importance and score of the included features
print(my_tree_one.score(features_one, target))

test.Fare[152] = test['Fare'].median()    # Impute the missing value with the median
test_features = test[['Pclass', 'Sex', 'Age', 'Fare']].values
test["Sex"][train["Sex"] == "male"] = 0
test['Sex'][train['Sex'] == 'female'] = 1
my_prediction = my_tree_one.predict(test_features)
print(my_prediction)

PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
print(my_solution)
print(my_solution.shape)
my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])

# Random forest analysis
features_two = train[["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]].values
max_depth = 10    #Control overfitting
min_samples_split = 5
my_tree_two = tree.DecisionTreeClassifier(max_depth = max_depth, min_samples_split = min_samples_split, random_state = 1)
my_tree_two = my_tree_two.fit(features_two, target)
print(my_tree_two.score(features_two, target))

train_two = train.copy()
train_two["family_size"] = train_two["SibSp"] + train_two["Parch"] + 1    # Create family_size feature
features_three = train_two[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "family_size"]].values
my_tree_three = tree.DecisionTreeClassifier()
my_tree_three = my_tree_three.fit(features_three, target)
print(my_tree_three.score(features_three, target))
features_forest = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)
my_forest = forest.fit(features_forest, target)
print(my_forest.score(features_forest, target))

test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
pred_forest = my_forest.predict(test_features)
print(len(pred_forest))
print(my_tree_two.feature_importances_)
print(my_forest.feature_importances_)
print(my_tree_two.score(features_two, target))
print(my_forest.score(features_forest, target))
