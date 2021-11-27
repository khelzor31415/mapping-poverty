import os
import shutil
import operator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
#from google.cloud import storage

import seaborn as sns
from sklearn import preprocessing
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy.stats import percentileofscore
from sklearn.mixture import GaussianMixture
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

TM_pal_categorical_3 = ("#ef4631", "#10b9ce", "#ff9138")
sns.set(
    style="white",
    font_scale=1,
    palette=TM_pal_categorical_3,
)

SEED = 42
np.random.seed(SEED)

def plot_hist(data, title, x_label, y_label, bins=30):
	"""
	Plots histogram
	
	Parameters
	----------
	data: pandas Series
		Data to plot histogram
	title : str
		The title of the figure
	x_label : str
		Label of the x-axis
	y_label : str
		Label of the y-axis
	bins : int
		Number of bins for the histogram
	"""
	
	plt.hist(data, bins=bins)
	plt.title(title)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.show()

def plot_regplot(
	data,
	x_label,
	y_label,
	x_var,
	y_var
):
	"""
	Produces the regression plot for the data
	
	Parameters
	----------
	data : pandas Series
		The data to plot regression plot
	x_var : str
		The variable name of the independent variable
	y_var : str
		The variable name of the dependent variable
	x_label : str
		The label of the x-axis
	y_label : str
		The label of the y-axis
	"""
	
	ax = sns.regplot(
		x=x_var,
		y=y_var,
		data=data,
		lowess=True,
		line_kws={'color':'black', 'lw':2},
		scatter_kws={'alpha':0.3}
	)
	
	plt.ticklabel_format(style='sci', axis='x', scilimits=(1,5))
	plt.title(
		"Relationship between {}\nand {}".format(
			x_label, y_label
		)\
		+ r"($\rho$ = %.2f, $r$ = %.2f)"
		% (
			spearmanr(
				data[x_var].tolist(), data[y_var].tolist()
			)[0],
			pearsonr(
				data[x_var].tolist(), data[y_var].tolist()
			)[0]
		)	
	)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.show()

def plot_corr(
	data,
	features_cols,
	indicator,
	figsize,
	max_n
):
	"""
	Produces a barplot of the Spearman rank correlation and Pearson's correlation for a group of values in descending order
	
	Parameters
	----------
	data: pandas DataFrame
		The DataFrame containing the features columns
	features_cols : list of str
		The list of feature column names in the data
	indicator: str
		The socioeconimic indicator to correlate each variable with
	figsize: tuple of ints
		Size of the figure
	max_n : int
		Maximum number of variables to plot
	"""
	
	n = len(features_cols)
	spearman, pearsons = [], []
	
	for feature in features_cols:
		spearman.append(
			(
				feature,
				spearmanr(data[feature], data[indicator])[0]
			)
		)
		pearsons.append(
			(
				feature,
				pearsonr(data[feature], data[indicator])[0]
			)
		)
	spearman = sorted(spearman, key=lambda x:abs(x[1]))
	pearsons = sorted(pearsons, key=lambda x:abs(x[1]))

	plt.figure(figsize=figsize)
	plt.title(
		"Spearman Correlation Coefficient for {}".format(indicator)
	)
	plt.barh(
		[x[0] for x in spearman[n - max_n:]],
		[x[1] for x in spearman[n - max_n:]]	
	)
	
	plt.grid()

	plt.figure(figsize=figsize)
	plt.title(
		'Pearsons Correlation Coefficient for {}'.format(indicator)
	)
	plt.barh(
		[x[0] for x in pearsons[n - max_n:]],
		[x[1] for x in pearsons[n - max_n:]]
	)
	plt.grid()
		

