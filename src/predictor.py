
import numpy, sys, warnings, os, time
from src import db_connector
from src.constants import signal_features_names
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectPercentile, f_regression, mutual_info_regression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet


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


def run_predictor():

    RANDOM_STATE = 2401

    qc_features_database_path = "/Users/andreidm/ETH/projects/monitoring_system/res/nas2/qc_features_database.sqlite"
    features_meta, features, features_names = get_features_data(qc_features_database_path)

    signal_features_indices = [features_names.index(feature)-4 for feature in signal_features_names]
    signal_features = features[:, numpy.array(signal_features_indices)]

    # impute column-wise with median
    for i in range(signal_features.shape[1]):
        signal_features[numpy.where(signal_features[:, i] == -1), i] = numpy.median(signal_features[:, i])

    # fit for each feature
    for i in range(signal_features.shape[1]):

        X = numpy.delete(signal_features, i, axis=1)
        y = signal_features[:, i]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)

        pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('selector', SelectPercentile()),
            ('model', ElasticNet(max_iter=5000, random_state=RANDOM_STATE))
        ])

        parameters = {
            'selector__score_func': [f_regression, mutual_info_regression],
            'selector__percentile': [x for x in range(5, 105, 10)],

            'model__alpha': [5e-05, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5.0, 10, 50.0, 100, 500.0, 1000],
            'model__l1_ratio': [x / 10. for x in range(1, 11, 1)],
            'model__fit_intercept': [True, False]
        }

        start = time.time()
        print("Fitting models for {}...".format(signal_features_names[i]))
        grid = GridSearchCV(pipeline, parameters, scoring='neg_median_absolute_error', cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)

        print("Best parameter CV scores:", grid.best_score_)
        print("Mean target value:", numpy.mean(y))
        print("Parameters:", grid.best_params_)
        print((time.time() - start) // 60, 'minutes elapsed\n')


if __name__ == '__main__':

    # HARD SUPPRESS OF ALL WARNINGS
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"

    run_predictor()