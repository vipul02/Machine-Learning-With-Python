from sklearn import tree

# [height, weight, shoe]
x = [[180, 80, 40], [170, 70, 38], [160, 60, 37], [160, 70, 38],
    [170, 80, 42], [160, 80, 38], [150, 50, 36], [155, 60, 38], 
    [180, 50, 40], [185, 50, 42], [160, 50, 40], [165, 60, 38]]

y = ['male', 'female', 'female', 'female', 'male', 'male',
    'female', 'female', 'male', 'male', 'female', 'female']

clf = tree.DecisionTreeClassifier()
clf = clf.fit(x, y)
prediction = clf.predict([[180, 70, 39]])
print(prediction)
# todo: apply predictio to this dataset using different models and give the model which gives max accuracy