Eshaan Gandhi

Breast Cancer Classification Tool

**Overview**

I recreated a model that makes a diagnosis of whether someone has breast cancer or not looking at a certain number of characteristics. The Diagnostic Wisconsin Breast Cancer Database donated the data set I used. There are some more details about the data set in this document. The program can predict the occurrence of Breast Cancer with an accuracy of ~94%. I will explain my process, code, and learnings in this document.

**Data Set**

Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe the characteristics of the cell nuclei present in the image.

This database is also available through the UW CS ftp server:

ftp ftp.cs.wisc.edu

cd math-prog/cpo-dataset/machine-learn/WDBC/

Also can be found on the UCI Machine Learning Repository: [https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)

Attribute Information:

1) ID number

2) Diagnosis (M = malignant, B = benign)

3-32)

Ten real-valued features are computed for each cell nucleus:

a) radius (mean of distances from the center to points on the perimeter)

b) texture (standard deviation of gray-scale values)

c) perimeter

d) area

e) smoothness (local variation in radius lengths)

f) compactness (perimeter^2 / area - 1.0)

g) concavity (severity of concave portions of the contour)

h) concave points (number of concave portions of the contour)

i) symmetry

j) fractal dimension (&quot;coastline approximation&quot; - 1)

The mean, standard error, and &quot;worst&quot; or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features. For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius. All feature values are recoded with four significant digits. Missing attribute values: none. Class distribution: 357 benign, 212 malignant

**Method**

I first cleaned the data. There are a lot of Python functions that worked here, and this was a relatively easy step. I then used the first 500 entries in the data set to train my algorithm. The algorithm that I used was logistic Regression using gradient descent. Below is an explanation of what logistic regression and gradient descent is.

**Logistic Regression**

Logistic Regression is the appropriate regression analysis to conduct when the dependent variable is dichotomous (binary). Like all regression analyses, logistic regression is a predictive analysis. Logistic Regression is used to describe data and to explain the relationship between one dependent binary variable and one or more nominal, ordinal, interval, or ratio-level independent variables. Logistic Regression is a classification algorithm that gives you a binary output. They take in several inputs. Logistic Regression is used when you need a yes/no answer. For example, if one has to predict if an email is spam or not, if a person in the U.S. owns a U.S.-made or foreign-made (non-U.S.) car, if one is sick or not based on specific characteristics.

The logistic regression was implemented using Gradient Descent. It works in the following way. We take partial derivatives of the cost function concerning every weight variable in the network then slightly adjust them in the direction that would minimize the cost.

This algorithm approaches the local minimum rapidly at first but then slows down once it has gotten close.

**Testing**

The data set consists of more than 500 entries. I split the entries and used one set to train the Artificial Intelligence and the other set to test it. Also, if you look at the dataset, Benign and Malignant Datasets are clumped together. Hence, I shuffled them to remove the bias from the data set. This significantly increased the accuracy of the AI by as much as 10%. I have also made an option where the user can test their inputs. There are test cases enclosed in the folder that I submitted.
