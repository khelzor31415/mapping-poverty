from scipy import stats 
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor

def evaluate_model():
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

        # insert cross validation using param grid here

        for score in nested_scores:
            print(score, ': %.4f' % nested_scores[score].mean())
            print(nested_scores[score])
            if score == 'test_r2':
                r_squared = nested_scores[score].mean()
        formatted_indicator = ' '.join([x for x in indicator.split() if '(' not in x]).title() # ??? huh
    
        cv.fit(X, y)
        print('Best estimator : {}'.format(cv.best_estimator_))
    
    return pd.DataFrame(results)

    results[indicator + '_pred'] = y_pred
    results[indicator + '_true'] = y_true
    results[clust_str] = clusters
def get_param_grid(model_type):
    if model_type == 'lasso':
        param_grid = {
            'regressor__alpha' : stats.uniform.rvs(loc=0, scale=10, size=1000),
            'regressor__normalize' : [True, False]
        }
    elif model_type == 'elastic_net':
        param_grid = {
            'regressor__alpha' : stats.uniform.rvs(loc=0, scale=10, size=1000), 
            'regressor__l1_ratio' : np.random.uniform(0, 1, 1000)
            'regressor__normalize' : [True, False]
        }
    elif model_type == 'random_forest':
        param_grid = {
            'regressor__n_estimators' : stats.randint(200, 2000, 1000), 
            'regressor__criterion' : ['squared_error'], 
            'regressor__max_features' : ['auto', 'sqrt', 'log2'], 
            'regressor__max_depth' : stats.randint.rvs(3, 20, 10), 
            'regressor__min_samples_split' : stats.randint.rvs(2, 20, 10), 
            'regressor__min_samples_leaf' : stats.randint.rvs(1, 20, 10), 
            'regressor__bootstrap' : [True, False]
        }
    
    return param_grid
def get_model(model_type):
    if model_type == 'lasso':
        model = Lasso(random_state=SEED)
    elif model_type == 'elastic_net':
        model = ElasticNet(random_state=SEED)
    elif model_type == 'random_forest':
        model = RandomForestRegressor(random_state=SEED, n_jobs=-1)

    return model