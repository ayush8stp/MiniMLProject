#using pandas for data handling
import pandas as pd
#t0 make a basic classifier
from sklearn.naive_bayes import GaussianNB
#to split the data for training and testing
from sklearn.model_selection import train_test_split
#for checking accuracy of model
from sklearn.metrics import accuracy_score
#visualizing data as matplotlib
import matplotlib.pyplot as plt


#getting the data from the csv type of file
filename = open("Iris.csv" ,'r')
dataset = pd.read_csv(filename)

#Giving a numerical value to the class/type given of flower
ClassType = {'Iris-setosa': 1,'Iris-versicolor': 2 ,'Iris-virginica': 3}
dataset.Species = [ClassType[item] for item in dataset.Species]

#making the X values ,the independent features
x = dataset.drop('Species',axis=1)
#making the Y values, the dependent values
y = dataset['Species']

#splitting the data with default parameter
Xtrain , Xtest ,Ytrain ,Ytest = train_test_split(x ,y)

#making the classifier
reg = GaussianNB()
#fitting the classifier with training data
reg.fit(Xtrain ,Ytrain)
#now predicting the result for test cases
pred = reg.predict(Xtest)

#printing the result of test data
for p in pred:
    if p==1:
        print('Iris-setosa')
    if p==2:
        print('Iris-versicolor')
    if p==3:
        print('Iris-virginica')

#printing the accuracy score
print(accuracy_score(Ytest , pred))


#plotting the data for visual purpose
plt.scatter(Xtrain['SepalLengthCm'] ,Ytrain , color ='b' ,label='Training data')
plt.scatter(Xtest['SepalLengthCm'] ,Ytest ,color ='r' ,label = 'Testing Data')
plt.plot(Xtest['SepalLengthCm'] , pred , 'ro')
plt.show()