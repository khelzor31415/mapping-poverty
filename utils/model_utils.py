from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import random
import seaborn as sns
from scipy import stats
from math import sqrt

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline 
import xgboost as xgb

from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    KFold,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
    train_test_split,
    cross_val_predict
)
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import(
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor
)
from sklearn.linear_model import(
    LinearRegression,
    LogisticRegression,
    Ridge,
    Lasso,
    ElasticNet
)
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    PolynomialFeatures
)
from sklearn.metrics import (
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    make_scorer,
    roc_auc_score,
    balanced_accuracy_score
)
from sklearn.svm import SVR, SVC

SEED = 42
np.random.seed(SEED)




def get_param_grid(model_type):
    """
    Returns the model parameter grid to be used as an input for cross validation hyper parameter optimization

    Parameters
    ----------
    model_type : str
        type of model to use ; implemented models are : 'ridge', 'random_forest', and 'xgboost'
    
    Returns
    -------
    dict
        a dictionary of parameters
    """

    if (model_type == 'ridge') or (model_type == 'lasso'):
        param_grid = {
            'regressor__alpha' : stats.uniform.rvs(loc=0, scale=4, size=4),
            'regressor__normalize' : [True, False]
        }
    if model_type == 'elastic_net':
        param_grid = {
            'regressor__alpha' : stats.uniform.rvs(loc=0, scale=4, size=4),
            'regressor__l1_ratio' : np.random.uniform(0, 1, 10),
            'regressor__normalize' : [True, False]
        }
    elif model_type == 'random_forest':
        param_grid = {
            'regressor__n_estimators' : stats.randint(200,2000), # stats.randint(200, 2000)
            'regressor__max_features' : ['auto', 'sqrt', 'log2'],
            'regressor__max_depth' : stats.randint(3, 10), # stats.randint(3, 10)
            'regressor__min_samples_split' : stats.randint(2, 10), # stats.randint(2, 10)
            'regressor__min_samples_leaf' : stats.randint(1, 10), # stats.randint(1, 10)
            'regressor__bootstrap' : [True, False] 
        }
    elif model_type == 'xgboost':
        param_grid = {
            'regressor__n_estimators' : stats.randint(200, 2000),
            'regressor__learning_rate' : np.random.uniform(1e-3, 0.2, 1000),
            'regressor__subsample' : np.random.uniform(0.9, 1, 1000),
            'regressor__max_depth' : stats.randint(3, 10),
            'regressor__colsample_bytree' : np.random.uniform(0.7, 1, 100),
            'regressor__min_child_weight' : stats.randint(1, 5),
            'regressor__gamma' : np.random.uniform(0.5, 5, 100)
        }
    elif model_type == 'svr':
        param_grid = {
            'regressor__kernel' : ['linear', 'poly', 'rbf'],
            'regressor__degree' : stats.randint(1, 5),
            'regressor__gamma' : ['auto', 'scale'],
            'regressor__C' : np.random.uniform(0, 10, 100)
        }
    
    return param_grid

def get_model(model_type):
    """
    Returns the model instance to be used for cross validation

    Parameters
    ----------
    model_type : str
        type of model to use ; implemented models are 'ridge', 'random_forest', and 'xgboost'
    
    Returns
    -------
    model instance
        model to be evaluated
    """

    if model_type == 'ridge':
        model = Ridge(random_state=SEED)
    elif model_type == 'lasso':
        model = Lasso(random_state=SEED)
    elif model_type == 'elastic_net':
        model = ElasticNet(random_state=SEED)
    elif model_type == 'random_forest':
        model = RandomForestRegressor(random_state=SEED, n_jobs=-1)
    elif model_type == 'xgboost':
        model = xgb.XGBRegressor(
            objective='reg:linear',
            random_state=SEED,
            n_jobs=-1
        )
    elif model_type == 'svr':
        model = SVR()
    return model

def evaluate_model(
    data,
    feature_cols,
    indicator_cols,
    wandb,
    model_type,
    scoring={'r2':'r2'},
    refit='r2',
    search_type='random',
    n_splits=5,
    n_iter=50,
    task_type='regression',
    plot_importance=False,
    figsize=(5,8),
    std_scale=False,
    minmax_scale=False,
    polynomial=False,
    poly_degree=2,
    n_workers=-1,
    verbose=0,
    plot=True
):
    """
    Trains and evaluates the specified model on a given dataset using an n-fold nested cross validation scheme - implemented are ridge regression, random forest regression, and xgboost regression

    Parameters
    ----------
    data : pandas DataFrame
        a DataFrame containing the features and indicators columns
    feature_cols : list
        list of predictive features to be used for the training
    indicator_cols : list
        list of indicators to be predicted
    model_type : str
        type of model to use - implemented models are 'ridge', 'random_forest', and 'xgboost'
    scoring : dict
        dictionary containing the scoring metrics to be used
    refit : str
        scoring metric to be optimized
    search_type : str
        either 'grid' or 'random'
    n_splits : int
        number of splits/folds for the n-fold cross validation
    n_iter : int
        number of iterations for the random search cross validation
    task_type : str
        either 'classification' or 'regression'
    plot_importance : bool
        whether or not to plot importance; applicable only to random forest and xgboost
    figsize : tuple
        size of feature importance plot
    std_scale : bool
        indicate whether or not to apply standard scaling to features
    minmax_scale : bool
        indicates whether or not to apply min-max scaling to features

    Returns
    -------
    pandas DataFrame
        contains cluster number, actual, and predicted socioeconomic features
    """

    clust_str = 'Cluster number'
    param_grid = get_param_grid(model_type)

    results = {
        indicator + type_: []
        for indicator in indicator_cols 
        for type_ in ['_pred', '_true']
    }

    for index, indicator in enumerate(indicator_cols):
        X = data[feature_cols]
        y = data[indicator].tolist()
        clusters = data[clust_str].tolist()
        model = get_model(model_type)
        
        cv, nested_scores, y_true, y_pred = nested_cross_validation(
            model,
            X,
            y,
            param_grid=param_grid,
            scoring=scoring,
            refit=refit,
            search_type=search_type,
            n_splits=n_splits,
            n_iter=n_iter,
            task_type=task_type,
            std_scale=std_scale,
            minmax_scale=minmax_scale,
            polynomial=polynomial,
            poly_degree=poly_degree,
            n_workers=n_workers,
            verbose=verbose
        )

        print('Socioeconomic Indicator: {}'.format(indicator))
        for score in nested_scores:
            print(score, ': %.4f' % nested_scores[score].mean())
            print(nested_scores[score])
            if score == 'test_r2':
                r_squared = nested_scores[score].mean() # mukhang may mali rito
        
        formatted_indicator = ' '.join([x for x in indicator.split() if '(' not in x]).title()
        
        if wandb is not None:
            wandb.log({'{} R-squared'.format(formatted_indicator) : r_squared})
        
        if plot:
            plot_cross_val_results(
                y_true,
                y_pred,
                formatted_indicator,
                nested_scores,
                wandb=wandb,
                refit=refit
            )
        
        cv.fit(X, y)
        print('Best estimator : {}'.format(cv.best_estimator_))

        results[indicator + '_pred'] = y_pred
        results[indicator + '_true'] = y_true 
        results[clust_str] = clusters

        if plot_importance:
            if model_type == 'random forest':
                rf_feature_importance(cv, X, y, size=figsize)
            elif model_type == 'xgboost':
                cv, X, y, size=figsize
    return pd.DataFrame(results)


def nested_cross_validation(
    model,
    X, 
    y, 
    param_grid,
    scoring={"r2": "r2"}, 
    refit='r2',
    search_type='random',
    n_splits=5,
    n_iter=50,
    std_scale=False,
    minmax_scale=False,
    polynomial=False,
    poly_degree=2,
    task_type='regression',
    n_workers=-1,
    verbose=0
):
    if task_type == 'classification':
        inner_cv = StratifiedKFold(
            n_splits = n_splits,
            shuffle = True,
            random_state = SEED
        )
        outer_cv = StratifiedKFold(
            n_splits = n_splits, 
            shuffle = True,
            random_state = SEED
        )
    elif task_type == 'regression':
        inner_cv = KFold(
            n_splits = n_splits, 
            shuffle = True,
            random_state = SEED
        )
        outer_cv = KFold(
            n_splits = n_splits,
            shuffle = True, 
            random_state = SEED
        )
    pipeline = []
    if std_scale:
        std_scaler = StandardScaler()
        pipeline.append(('std_scale', std_scaler))
    if minmax_scale:
        minmax_scaler = MinMaxScaler()
        pipeline.append(('minmax_scale', minmax_scaler))
    if polynomial:
        poly = PolynomialFeatures(degree=poly_degree)
        pipeline.append(('poly', poly))
    if task_type == 'regression':
        pipeline.append(('regressor', model))
    
    pipe = Pipeline(pipeline)

    if search_type == 'grid':
        cv = GridSearchCV(
            estimator = pipe, 
            scoring = scoring, 
            param_grid = param_grid, 
            cv = inner_cv, 
            verbose = verbose, 
            n_jobs = n_workers, 
            refit = refit
        )
    elif search_type == 'random':
        cv = RandomizedSearchCV(
            estimator = pipe, 
            n_iter = n_iter, 
            scoring = scoring, 
            param_distributions = param_grid, 
            cv = inner_cv, 
            verbose = verbose,
            random_state = SEED, 
            n_jobs = n_workers, 
            refit = refit
        )
    

    nested_scores = cross_validate(
        cv,
        X = X,
        y = y,
        cv = outer_cv, 
        n_jobs = n_workers,
        scoring = scoring, 
        verbose = verbose, 
        return_train_score = True
    )

    y_pred = cross_val_predict(
        cv, 
        X = X,
        y = y, 
        cv = outer_cv, 
        n_jobs = n_workers, 
        verbose = verbose
    )

    return cv, nested_scores, y, y_pred

def plot_cross_val_results(
    y_true,
    y_pred,
    indicator,
    nested_scores,
    model_type,
    refit='r2'
):
    ax = sns.regplot(
        y_true,
        y_pred,
        line_kws={'color':'black', 'lw':1},
        scatter_kws={'alpha':0.3}
    )
    plt.title(
        indicator+r" $r^2: {0:.3f}$".format(
            nested_scores['test_' + str(refit)].mean()
        )
    )
    
    plt.xlabel('Observed wealth index')
    plt.ylabel('Predicted wealth index')
    plt.setp(ax.get_xticklabels(), rotation=10)
    plt.setp(ax.get_yticklabels(), rotation=10)
    plt.tight_layout()
    plt.savefig('../images/final model/ml_{}_osmnetcols.png'.format(indicator), transparent=True, pad_inches=0, dpi=300)
    plt.show()

def random_forest_feature_importance(
    cv, 
    X, y,
    n_features=30,
    size=(10,15)
):
    feat_impt = {}
    for z in range(len(X.columns)):
        feat_impt[
            X.columns[z]
        ] = cv.best_estimator_.named_steps[
            'regressor'
        ].feature_importances_[
            z
        ]
    pd.DataFrame(
        {'Feature Importance' : feat_impt}
    ).sort_values(
        by='Feature Importance', ascending=False
    ).iloc[
        :n_features
    ].plot(
        kind='barh', figsize=size
    )

    plt.grid()
    plt.gca().invert_yaxis()
    plt.show()