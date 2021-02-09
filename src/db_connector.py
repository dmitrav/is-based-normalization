import sqlite3, pandas


def create_connection(db_file):
    """ Creates a database connection to the SQLite database specified by db_file. """

    db = None
    try:
        db = sqlite3.connect(db_file)
        return db
    except Exception as e:
        print(e)

    return db


def fetch_table(conn, table_name):
    """ Gets data from the table_name given connection. """
    cur = conn.cursor()
    cur.execute("SELECT * FROM " + table_name)
    colnames = [description[0] for description in cur.description]

    return cur.fetchall(), colnames