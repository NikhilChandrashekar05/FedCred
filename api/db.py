#preforms the database connection for API Layer
#Reads from the same SQLite file that strategy.py writes too during training

import sqlite3
import os

# path to the database file created by strategy.py during federated training
DB_PATH = os.path.join(os.path.dirname(__file__), 'fedcredit.db')

def get_db():
    # open a connection to the SQLite database
    # check_same_thread=False allows FastAPI to use the connection across threads
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)

    # return rows as dictionaries instead of tuples
    # so you can access values by column name like row['accuracy']
    # instead of by index like row[2]
    conn.row_factory = sqlite3.Row

    return conn