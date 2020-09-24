
import warnings
warnings.filterwarnings("ignore")
import sys
sys.tracebacklimit=0
import os
import numpy as np
import random
import re

lim=600 #this is a placeholder integer that decides the number of input to be teaine

path = os.path.dirname(__file__)
file = os.path.join(path, 'wdbc.data')
file_object = open(file, 'r')

#Split dataset into separate points (as strings)
string_points = file_object.read().split('\n')
string_points.pop(-1)
random.shuffle(string_points) #Randomize (avoid bias)
num = 30

#Initialize dataset class arrays
point_array = np.empty((0,num))
y_labels = np.empty(0) #creating a array using the numPy library

#Trim data points
#Format as np.arrays
#Add to class arrays (Benign or Malignant)
for point in string_points:
    #print(point)
    point = point.split(',')
    #print(y_labels)
    if 'M' in point: #if malignant
        y_labels = np.append(y_labels, [1], axis=0)
    else: #if benign (can only be labeled 'B' or 'M')
        y_labels = np.append(y_labels, [0], axis=0)
        
    point = point[2:] #trim for only the important features
    #print(point)
    temp = np.array(point) #convert to numpy array
    #print(temp)
    temp = temp.astype(float) #cast as float array
    point_array = np.append(point_array, [temp], axis=0)
    #print(point_array)

#Split training and testing data
experience_matrix = point_array[0:500] #the first 400 hundred entries are the experience set
experience_matrix_y = y_labels[0:500] 
test_matrix = point_array[500:] #the last 400 are the test ex
test_matrix_y = y_labels[500:]

#trying to implement sklearn and boom that's it

from sklearn.linear_model import LogisticRegression
# all parameters not specified are set to their defaults
logisticRegr = LogisticRegression(random_state=0)
logisticRegr.fit(experience_matrix, experience_matrix_y);

Y_pred = logisticRegr.predict(test_matrix)

print("Hello user. You have two choices. Feed in your own data or just test the accuracy.")
print("To enter your own data press 1 and then enter")
print("To just test accuracy and look at the test data set press 2 and then enter")
o=input("Put in your choice\n");      
if(o=='1'):
    option=input("Enter the file name that contains the data to be run\n")
    path = os.path.dirname(__file__)
    file = os.path.join(path, option)
    file_object = open(file, 'r')
    ex_matrix_y=np.empty(0)
    final_y=np.empty(0)
    #Split dataset into separate points (as strings)
    spoints = file_object.read().split('\n')
    ex_matrix_y=spoints[0].split(',')
    for i in range(len(ex_matrix_y)):
        t = float(ex_matrix_y[i])
        #print(t)
        final_y=np.append(final_y,[t], axis=0)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(test_matrix_y, Y_pred)
    y_f=[final_y];
    y=logisticRegr.predict(y_f)

    if(y[0]==1):
        print("The tumor is Malignant. You should consult a doctor")
    else:
        print("The tumor is Benign. You are ~90% safe. Still consult a doctor if you feel like it.")
elif(o=='2'):
    #print(final_y)
    y_pred_l=len(Y_pred)
    count=0
    for i in range(y_pred_l):
        if Y_pred[i]==test_matrix_y[i]:
            count+=1
    
    accuracy=count/y_pred_l
    print("The test data set is as follows")
    print(test_matrix)
    print("The test result matrix is as follows")
    print("1 means Malignant and 0 means Benign")
    print(test_matrix_y)
    print("The accuracy in which this bot does the classification is")
    accuracy=round(accuracy*100,2)
    #str(accuracy)
    print((str)(accuracy)+"%")

else:
    print("Please enter only 1 or 2. Now rerun me if you want to test me")
