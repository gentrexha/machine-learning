## 1. Description

This is an implementation of some ML classification algorithms on given
datasets using the scikit learn library.

Used classifiers:

* K-Nearest Neighbors
* Bayesian Networks
* Decision Trees
* Random forests

## 2. Setup

Install needed packages by running (not sure whether this works on
windows):


```
pip install -r requirements.txt
```


## 3. Data preprocessing

All preprocessing scripts (if needed) are under the `preprocessors`
directory.

Running preprocessor scripts:

```
python preprocessors/bank.py
```

### 3.1 Bank Dataset

* Perform one-hot encoding for job, marital, contact columns
* Perform integer encoding for following:
    * Education (unknown 0, and then primary, secondary and tertiary)
    * Month (1-12 as normally)
    * poutcome (which means outcome of previous campaign)
        * failure 0, unknown and other 1, success 2 (put emphasis on the
        distance between failure and success)


#### 3.1.1 Open questions

* Consider whether it makes sense to do integer encoding for job
column. Some jobs higher rated than the others?!
* Maybe also reconsider whether the one-hot-encoded columns make sense


### 3.2 Breast Cancer


### 3.3 KDD Cup 98

This dataset contains about  445 features, which means that probably
a bunch of them should not be used by our model at all.
More information describing the dataset and the columns can be found
here: https://kdd.ics.uci.edu/databases/kddcup98/epsilon_mirror/cup98dic.txt

* Fill every numerical feature with the mean of the column and the nominal values 
with the most often used value.



#### 3.3.2 Feature Selection Methods

* Useful links:

https://towardsdatascience.com/why-how-and-when-to-apply-feature-selection-e9c69adfabf2

https://towardsdatascience.com/a-feature-selection-tool-for-machine-learning-in-python-b64dd23710f0

##### 3.3.2.1 Remove all nominal values
For exercise reasons I'll try to remove all nominal values and train models only based
 on the numerical values in the dataset and considering most nominal values have low
 variance this might actually somehow work.
 
 
##### 3.3.2.2 Dummy Method
Here we basically just referr to a book which is working with the same dataset
and look how the models are performing based on the selection of features they made.

See http://summit.sfu.ca/item/6396 for more info.
##### 3.3.2.3 Filter Methods
~ To be researched.
##### 3.3.2.4 Wrapper Methods, and
~ To be researched.
##### 3.3.2.5 Embedded Methods.
~ To be researched.

## 4. Procedure

Each usecase is implemented as separate script within `classifiers`
directory.

Datasets are converted pandas dataframes and then passed to scikit
classifiers

Running classifiers:

```
python classifiers/k_nearest_neighbors.py --dataset_name bank_marketing --n_neighbors 10 --weight uniform
```

or just change default click parameters on according scripts
