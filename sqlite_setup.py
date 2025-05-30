# import required libraries
import sqlite3
import pandas as pd
from langsmith import traceable  

@traceable(name="get_sqlite_conn")
def get_sqlite_conn(csv_path="data/real_estate_data.csv", db_path="real_estate.db"):
    df = pd.read_csv(csv_path)
    conn = sqlite3.connect(db_path)

    # Write DataFrame to SQLite
    df.to_sql("real_estate", conn, if_exists="replace", index=False)

    return conn
