# Preprocessing

Do the feature correlation and remove the ones with high correlation
(since they do not contribute any good o the overall model)

* Upload a png of the plot, so that it can be explained/visualized which
    are correlated or not
	* Consider the SelectKBest from https://machinelearningmastery.com/feature-selection-machine-learning-python/
	* Decision trees with feature importance (also upload the feature importance bar)
	* Remove features with correlation greater than a threshold (usually
	0.7)

* Find a way to export data from grid search automatically (GridSearchCV
and RandomizedSearchCV have a cv_results_ attribute, which can be used
to be imported to Pandas DF and then to csv for our report issues)


# TODO

* Store correlation heatmaps for all datasets
* store feature importances
* store grid searches names correspondingly (done)

Create jobs for all the datasets at the AC cluster
 * Bank marketing, Breast Cancer, Image Segmentation (raw data) - knn, naive bayes, DT, RF
    * All of the above, perform each classifier with grid_search
 * Perform step 1 in scaled, normalized, min_max_scaled data (in a non-feature selected dataset)

 * Find best features, and foreach apply classifications in step
    * k best features, feature importance non-zero, remove highly correlated
  * Perform in scaled, normalized data (with feature selection)
    * After cleaning data and selecting features, do the grid/random search (it makes more sense...maybe) (also, maybe
    we do not need it for step 1...does it make sense?)


    NOTE: Make sure to print as much as possible, so that you can have the
    job outputs. Also, make sure the naming of variables is really unique, and it does not
    overwrite each other

# Report generation ideas

* PARAMETER SELECTION: Create a bar graph with performance measures
    (accuracy, f-score, precision, recall). Foreach dataset, with each best parameters from
  random search cv , GridSearch CV and intuition (parameters we decided
  ourselves - make sure they are out of the interval you tried in grid/
  random search).

  Note: This can be generalized as Stardinate graph as well

* DATA PREPROCESSING: In order to show the role of preprocessing,
  feature selection and stuff, create a line graph for some of the performance measures
  (e.g. F1 score) with comparision like: raw data, dropped NA-s,
  numerical data, NA-s replaced with mode & mean, scaled, normalized,
  feature selection (drop highly correlated, selectKbest, drop features with
  zero importance)

