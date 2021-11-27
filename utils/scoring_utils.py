import os 
import shutil 
import operator

import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn import preprocessing
from scipy.stats import spearmanr, pearsonr, percentileofscore
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

SEED = 42
np.random.seed(SEED)

# SCORING FUNCTIONS

def pearsonr2(estimator, X, y_true):
	"""
	Calculates the r-squared score using the Pearson-R Test
	
	Parameters
	----------
	estimator
		The model or regresssor to be evaluated
	X : pandas DataFrame
		The feature matrix
	y : list of pandas Series
		The target vector
	
	Returns
	-------
	float
		r-squared value using Pearson-R test
	"""
	
	y_pred = estimator.predict(X)
	
	# READ UP ON THE DOCUMENTATION, WHY IS IT SQUARED
	return(pearsonr(y_true, y_pred))[0]**2 

def mae(estimator, X, y_true):
	"""
	Calculates mean absolute error
	
	Parameters
	----------
	estimator
		The model or regressor to be evaluated
	X : pandas DataFrame
		The feature matrix
	y : list of pandas Series
		The target vector
	
	Returns
	-------
	float
		Mean absolute error
	"""
	
	y_pred = estimator.predict(X)
	return mean_absolute_error(y_true, y_pred)

def rmse(estimator, X, y_true):
	"""
	Calculates root mean squared error
	
	Parameters
	----------
	estimator
		The model or regressor to be evaluated
	X : pandas DataFrame
		The feature matrix
	y : list of pandas Series
		The target vector
	Returns
	-------
	float
		Root mean squared error
	"""
	
	y_pred = estimator.predict(X)
	return np.sqrt(mean_squared_error(y_true, y_pred))

def r2(estimator, X, y_true):
	"""
	Calculates the r-squared score 
	
	Parameters
	----------
	estimator
		The model or regressor to be evaluated
	X : pandas DataFrame
		The feature matrix
	y : list of pandas Series
		The target vector
	
	Returns
	-------
	float
		R-squared score
	"""
	
	y_pred = estimator.predict(X)
	return r2_score(y_true, y_pred)

def mape(estimator, X, y_true):
	"""
	Calculates mean average percentage error
	
	Parameters
	----------
	estimator
		The model to be evaluated
	X : pandas DataFrame
		The feature matrix
	y : list of pandas Series
		The target vector

	Returns 
	-------
	float
		Mean average percentage error
	"""
	
	y_pred = estimator.predict(X)
	return np.mean(np.abs(y_true-y_pred) / np.abs(y_true)) * 100

def adj_r2(estimator, X, y_true):
	"""
	Calculates adusted r-squared score

	Parameters
	----------
	estimator
		The model to be evaluated
	X : pandas DataFrame
		The feature matrix
	y : list of pandas Series
		The target vector
	Returns
	-------
	float
		Adjusted r-squared score
	"""
	
	y_pred = estimator.predict(X)
	r2 = r2_score(y_true, y_pred)
	n = X.shape[0]
	k = X.shape[1]
	adj_r2 = 1 - (((1-r2)*(n-1))/(n - k - 1))

def percentile_ranking(series):
	"""
	Converts list of numbers to percentile and ranking

	Parameters
	----------
	series: pandas Series
		A series of numbers to be converted to percentile ranking
	Returns
	-------
	list of floats
		A list of converted percentile values using scipy.stats percentileofscore()
	list of ints
		A list containing the ranks
	"""
	
	percentiles = []
	for index, value in series.iteritems():
		curr_index = series.index.isin([index])
		percentile = percentileofscore(series[~curr_index], value)
		percentiles.append(percentile)
	ranks = series.rank(axis=0, ascending=False)

	return percentiles, ranks
	
	