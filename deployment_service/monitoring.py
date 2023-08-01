import json
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, Column, String, Text, Integer, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import sqlalchemy

Base = declarative_base()
class APILog(Base):
    __tablename__ = "api_logs"

    request_id = Column(String, primary_key=True)
    run_info = Column(Text)
    input_data = Column(Text)
    output_data = Column(Text)
    timestamp = Column(DateTime)
class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime)
    query_text = Column(Text)
    original_predictions = Column(Text)
    run_id = Column(String)
    host = Column(String)
    port = Column(Integer)
    feedback_labels = Column(Text)
class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_type = Column(String)
    experiment_name = Column(String)
    run_id = Column(String)
    host = Column(String)
    port = Column(Integer)
    status = Column(String)
    timestamp = Column(DateTime)
class DatabaseManagement:
    def __init__(self, db_uri):
        self.db_uri = db_uri
        print(f"Creating engine with db_uri: {self.db_uri}")
        self.engine = create_engine(self.db_uri)

    def create_tables(self):
        Base.metadata.create_all(self.engine)

    def get_active_models(self):
        with self.engine.connect() as connection:
            query = sqlalchemy.text("""
                SELECT DISTINCT e1.host, e1.port, e1.run_id
                FROM events e1
                JOIN (
                    SELECT run_id, host, port, MAX(id) as max_id
                    FROM events
                    GROUP BY run_id, host, port
                ) e2
                ON e1.run_id = e2.run_id AND e1.host = e2.host AND e1.port = e2.port AND e1.id = e2.max_id
                WHERE e1.status = 'active' AND e1.event_type != 'shutdown'
            """)
            df = pd.read_sql_query(query, connection)
        return df.to_dict('records')

class APIMonitoring:
    def __init__(self, db_management):
        self.db_management = db_management
        self.db_management.create_tables()

    def log_data(
            self, 
            request_id, 
            input_data, 
            output_data, 
            run_info):
        Session = sessionmaker(bind=self.db_management.engine)
        session = Session()

        log_entry = APILog(
            request_id=request_id,
            run_info=json.dumps(run_info),
            input_data=json.dumps(input_data),
            output_data=json.dumps(output_data),
            timestamp=datetime.now(),
        )
        session.add(log_entry)
        session.commit()

        session.close()
        print(f"Logged data to {self.db_management.db_uri}")

    def write_feedback_to_db(self, feedback_data):
        Session = sessionmaker(bind=self.db_management.engine)
        session = Session()

        feedback_entry = Feedback(
            query_text=feedback_data.query_text,
            original_predictions=json.dumps([dict(pred) for pred in feedback_data.original_predictions]),
            run_id=feedback_data.run_id,
            host=feedback_data.host,
            port=feedback_data.port,
            feedback_labels=",".join(feedback_data.feedback_labels),
            timestamp=datetime.now(),
        )

        session.add(feedback_entry)
        session.commit()
        session.close()

    def log_event(self, event_type, run_info, status):
        Session = sessionmaker(bind=self.db_management.engine)
        session = Session()

        event_entry = Event(
            event_type=event_type,
            run_id=run_info["run_id"],
            host=run_info["host"],
            port=run_info["port"],
            status=status,
            timestamp=datetime.now(),
        )
        session.add(event_entry)
        session.commit()

        session.close()
        print(f"Logged event to {self.db_management.db_uri}")



           
