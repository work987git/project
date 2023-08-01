import psycopg2
import pandas as pd

def create_database(conn):
    conn = psycopg2.connect(
    host="127.0.0.1",
    database="postgres",
    user="postgres",
    password="postgres"
    )
    print('ENTER1')
    conn.autocommit = True
    # SQL query to create the "events" table
    cur = conn.cursor()
    create_table_query = """
        CREATE TABLE IF NOT EXISTS events (
            id SERIAL PRIMARY KEY,
            event_type TEXT NOT NULL,
            model_name TEXT,
            experiment_name TEXT,
            run_id TEXT,
            host TEXT,
            port INTEGER,
            status TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    # Execute the query to create the table
    cur.execute(create_table_query)
    print('ENETER2')
    # Close t   he cursor and the connection
    cur.close()
    conn.close()

def create_feedback():
    conn = psycopg2.connect(
    host="127.0.0.1",
    database="postgres",
    user="postgres",
    password="postgres"
    )
    cur = conn.cursor()
    conn.autocommit = True
    create_feedback = """
        CREATE TABLE IF NOT EXISTS feedback (
            id BIGINT PRIMARY KEY,
            assessment_text TEXT NOT NULL,
            feedback TEXT,
            model_id VARCHAR
        )
    """
    cur.execute(create_feedback)
    conn.close()

def log_feedback(id, input_text, feedback, model_id):
    conn = psycopg2.connect(
    host="127.0.0.1",
    database="postgres",
    user="postgres",
    password="postgres"
    )
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute( """ INSERT INTO feedback (id, assessment_text, feedback, model_id)
    VALUES (%s, %s, %s, %s) """, 
    (id, input_text, feedback, model_id))
    conn.commit()
    conn.close()

def retrive_feedback(run_id):
    # run_id is model_id in the feedback table
    conn = psycopg2.connect(
    host="127.0.0.1",
    database="postgres",
    user="postgres",
    password="postgres"
    )
    conn.autocommit = True
    sql = 'SELECT * FROM feedback WHERE "run_id" = %s'
    feeds = pd.read_sql(sql = sql, con=conn, params=(run_id,))
    return feeds

def log_events(conn, event_type, model_info, status):
    conn = psycopg2.connect(
    host="127.0.0.1",
    database="postgres",
    user="postgres",
    password="postgres"
    )
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute( """ INSERT INTO events (event_type, model_name, experiment_name, run_id, host, port, status)
    VALUES (%s, %s, %s, %s, %s, %s, %s) """, 
    (event_type, model_info["model_name"], 
    model_info["experiment_name"], 
    model_info["run_id"], 
    model_info["host"], 
    model_info["port"], 
    status) )
    conn.commit()
    conn.close()