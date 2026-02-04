import pandas as pd
import sqlalchemy as sa
from sqlalchemy import text
import warnings
import os
import urllib.parse
from dotenv import load_dotenv  # pip install python-dotenv

load_dotenv()

def create_database_engine():
    """
    Create a database engine using environment variables.
    Stored in .env file (not in github repo)
    """
    
    # Get database configuration from environment
    db_type = os.getenv('DB_TYPE', 'postgresql')
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD')
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '5432')
    db_name = os.getenv('DB_NAME')
    
    # Validate required parameters
    if not all([db_user, db_password, db_name]):
        raise ValueError("Missing required database credentials in environment variables")
    
    # URL encode password to handle special characters
    encoded_password = urllib.parse.quote_plus(db_password)
    
    # Create connection string
    if db_type == 'postgresql':
        connection_string = f"postgresql://{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}"
    elif db_type == 'mysql':
        connection_string = f"mysql+pymysql://{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}"
    elif db_type == 'mssql':
        connection_string = f"mssql+pyodbc://{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}?driver=ODBC+Driver+17+for+SQL+Server"
    else:
        raise ValueError(f"Unsupported database type: {db_type}")
    
    # Create engine with production-ready settings
    engine = sa.create_engine(
        connection_string,
        # pool_size=10,              # Base number of connections in pool
        # max_overflow=20,           # Additional connections if needed
        # pool_timeout=30,           # Max seconds to wait for connection
        # pool_recycle=3600,         # Recycle connections after 1 hour
        # pool_pre_ping=True,        # Validate connections before use
        # echo=False#,                # Set True for debugging
        # connect_args={
        #     "connect_timeout": 10,  # Connection timeout in seconds
        #     "application_name": "MyDataApp"  # For PostgreSQL connection tracking
        # }
    )
    
    return engine

def get_reports_from_db(db_connection, reports_table_name, link_table_name):
    """
    Join reports table with link table using LEFT JOIN on 'eid' field.
    
    Args:
        db_connection: SQLAlchemy connection object or engine
        reports_table_name (str): Name of the reports table
        link_table_name (str): Name of the link table
    
    Returns:
        pandas.DataFrame: Joined data with all reports and matching links
    
    Raises:
        ValueError: If connection is None or invalid
        Warning: If tables don't exist or other issues are detected
    """
    
    # Validate database connection
    if db_connection is None:
        raise ValueError("Database connection cannot be None")
    
    try:
        # Test the connection
        if hasattr(db_connection, 'execute'):
            # It's a connection object
            conn = db_connection
        elif hasattr(db_connection, 'connect'):
            # It's an engine object
            conn = db_connection.connect()
        else:
            raise ValueError("Invalid database connection type. Expected SQLAlchemy Connection or Engine.")
        
        # Check if tables exist
        inspector = sa.inspect(db_connection if hasattr(db_connection, 'dialect') else db_connection.engine)
        existing_tables = inspector.get_table_names()
        
        if reports_table_name not in existing_tables:
            warnings.warn(f"Reports table '{reports_table_name}' does not exist in database")
            return pd.DataFrame()
        
        if link_table_name not in existing_tables:
            warnings.warn(f"Link table '{link_table_name}' does not exist in database")
            return pd.DataFrame()
        
        # Check if 'eid' column exists in both tables
        reports_columns = [col['name'] for col in inspector.get_columns(reports_table_name)]
        link_columns = [col['name'] for col in inspector.get_columns(link_table_name)]
        
        if 'eid' not in reports_columns:
            warnings.warn(f"Column 'eid' not found in reports table '{reports_table_name}'")
            return pd.DataFrame()
        
        if 'eid' not in link_columns:
            warnings.warn(f"Column 'eid' not found in link table '{link_table_name}'")
            return pd.DataFrame()
        
        # Construct the LEFT JOIN query
        # Using table aliases to handle potential column name conflicts
        query = text(f"""
            SELECT r.*, l.*
            FROM {reports_table_name} r
            LEFT JOIN {link_table_name} l ON r.eid = l.eid
        """)
        
        # Execute query and return DataFrame
        df = pd.read_sql(query, conn)
        
        # If we created a connection, close it
        if hasattr(db_connection, 'connect') and conn != db_connection:
            conn.close()
        
        return df
        
    except sa.exc.SQLAlchemyError as e:
        warnings.warn(f"Database error occurred: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        warnings.warn(f"Unexpected error occurred: {str(e)}")
        return pd.DataFrame()