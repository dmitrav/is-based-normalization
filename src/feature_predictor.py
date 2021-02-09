
import numpy
from src import db_connector
from src.constants import signal_features_names


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


if __name__ == '__main__':

    qc_features_database_path = "/Users/andreidm/ETH/projects/monitoring_system/res/nas2/qc_features_database.sqlite"
    features_meta, features, features_names = get_features_data(qc_features_database_path)

    signal_features_indices = [features_names.index(feature)-4 for feature in signal_features_names]
    signal_features = features[:, numpy.array(signal_features_indices)]


