
import numpy, sys, warnings, os, time, pandas
from tqdm import tqdm
from src import db_connector
from src.constants import signal_features_names
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectPercentile, f_regression, mutual_info_regression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet, Lasso, Ridge, BayesianRidge, Lars


def get_features_data(path):
    """ This method read metrics database,
        returns a matrix with metrics, metrics names, arrays of quality and acquisitions dates. """

    conn = db_connector.create_connection(path)
    if conn is None:
        raise ValueError("Database connection unsuccessful. Check out path. ")

    database_1, colnames_1 = db_connector.fetch_table(conn, "qc_features_1")
    database_2, colnames_2 = db_connector.fetch_table(conn, "qc_features_2")

    features_1 = numpy.array(database_1)
    features_2 = numpy.array(database_2)

    meta = features_1[:, :4]
    features = numpy.hstack([features_1[:, 4:].astype(numpy.float), features_2[:, 4:].astype(numpy.float)])
    colnames = [*colnames_1, *colnames_2[4:]]

    return meta, features, colnames


def get_models_and_parameters(random_state):

    models = {
        'elastic': ElasticNet(max_iter=5000, random_state=random_state),
        'lasso': Lasso(max_iter=5000, random_state=random_state),
        'ridge': Ridge(max_iter=5000, random_state=random_state),
        'bayes_ridge': BayesianRidge(n_iter=2000),
        'lars': Lars(random_state=random_state)
    }

    parameters = {

        'elastic': {
            'selector__score_func': [f_regression, mutual_info_regression],
            'selector__percentile': [5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 100],

            'model__alpha': [5e-05, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 200],
            'model__l1_ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.],
            'model__fit_intercept': [True, False]
        },

        'lasso': {
            'selector__score_func': [f_regression, mutual_info_regression],
            'selector__percentile': [10, 50, 100],

            'model__alpha': [5e-05, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 200],
            'model__fit_intercept': [True, False]
        },

        'ridge': {
            'selector__score_func': [f_regression, mutual_info_regression],
            'selector__percentile': [5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 100],

            'model__alpha': [5e-05, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 200],
            'model__fit_intercept': [True, False]
        },

        'bayes_ridge': {
            'selector__score_func': [f_regression, mutual_info_regression],
            'selector__percentile': [5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 100],

            'model__alpha_1': [1e-7, 1e-6, 1e-5],
            'model__alpha_2': [1e-7, 1e-6, 1e-5],
            'model__lambda_1': [1e-7, 1e-6, 1e-5],
            'model__lambda_2': [1e-7, 1e-6, 1e-5],
            'model__fit_intercept': [True, False]
        },

        'lars': {
            'selector__score_func': [f_regression, mutual_info_regression],
            'selector__percentile': [5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 100],

            'model__fit_intercept': [True, False]
        }
    }

    return models, parameters


def run_predictor(save_to):

    RANDOM_STATE = 2401

    qc_features_database_path = "/Users/andreidm/ETH/projects/monitoring_system/res/nas2/qc_features_database.sqlite"
    features_meta, features, features_names = get_features_data(qc_features_database_path)

    signal_features_indices = [features_names.index(feature)-4 for feature in signal_features_names]
    signal_features = features[:, numpy.array(signal_features_indices)]

    # impute column-wise with median
    for i in range(signal_features.shape[1]):
        signal_features[numpy.where(signal_features[:, i] == -1), i] = numpy.median(signal_features[:, i])

    models, parameters = get_models_and_parameters(RANDOM_STATE)
    models_names = sorted(list(models.keys()))  # being paranoid
    results = []

    # fit every feature
    for i in tqdm(range(signal_features.shape[1])):

        feature_preds = []
        # try every model
        for name in tqdm(models_names):

            X = numpy.delete(signal_features, i, axis=1)
            y = signal_features[:, i]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)

            pipeline = Pipeline([
                ('scaler', MinMaxScaler()),
                ('selector', SelectPercentile()),
                ('model', models[name])
            ])

            start = time.time()
            print("Fitting {} for {}...".format(name.upper(), signal_features_names[i]))
            grid = GridSearchCV(pipeline, parameters[name], scoring='neg_median_absolute_error', cv=3, n_jobs=-1)
            grid.fit(X_train, y_train)

            relative_error_percent = round(-grid.best_score_ / numpy.median(y), 4) * 100
            print("Best parameter CV scores:", grid.best_score_)
            print("Median target value:", numpy.median(y))
            print("Relative error: {}%".format(relative_error_percent ))
            print("Parameters:", grid.best_params_)
            print(int(time.time() - start), 'sec elapsed\n')

            feature_preds.append(relative_error_percent)

        results.append(feature_preds)

    results = pandas.DataFrame(results, columns=signal_features_names, index=models_names)
    results.to_csv(save_to + 'grid_search_results.csv')


if __name__ == '__main__':

    # HARD SUPPRESS OF ALL WARNINGS
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"

    save_to = '/Users/andreidm/ETH/projects/qc-based-normalization/res/'
    run_predictor(save_to)