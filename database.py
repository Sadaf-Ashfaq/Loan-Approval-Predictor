import sqlite3
import hashlib
import os
from datetime import datetime
import pandas as pd

DATABASE_NAME = 'loan_system.db'

def get_db_connection():
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_database():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS loan_applications (
            application_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            no_of_dependents INTEGER,
            education TEXT,
            self_employed TEXT,
            income_annum INTEGER,
            loan_amount INTEGER,
            loan_term INTEGER,
            cibil_score INTEGER,
            residential_assets_value INTEGER,
            commercial_assets_value INTEGER,
            luxury_assets_value INTEGER,
            bank_asset_value INTEGER,
            prediction TEXT,
            approval_probability REAL,
            rejection_probability REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_activity (
            activity_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            activity_type TEXT,
            activity_details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    ''')
    
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, email, password, full_name):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        password_hash = hash_password(password)
        
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, full_name)
            VALUES (?, ?, ?, ?)
        ''', (username, email, password_hash, full_name))
        
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return True, "Account created successfully!"
    except sqlite3.IntegrityError:
        return False, "Username or email already exists"
    except Exception as e:
        return False, f"Error: {str(e)}"

def verify_user(username, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    password_hash = hash_password(password)
    
    cursor.execute('''
        SELECT * FROM users WHERE username = ? AND password_hash = ?
    ''', (username, password_hash))
    
    user = cursor.fetchone()
    
    if user:
        cursor.execute('''
            UPDATE users SET last_login = ? WHERE user_id = ?
        ''', (datetime.now(), user['user_id']))
        conn.commit()
    
    conn.close()
    return dict(user) if user else None

def save_loan_application(user_id, input_data, prediction, approval_prob, rejection_prob):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO loan_applications (
            user_id, no_of_dependents, education, self_employed,
            income_annum, loan_amount, loan_term, cibil_score,
            residential_assets_value, commercial_assets_value,
            luxury_assets_value, bank_asset_value,
            prediction, approval_probability, rejection_probability
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        user_id,
        input_data['no_of_dependents'],
        input_data['education'],
        input_data['self_employed'],
        input_data['income_annum'],
        input_data['loan_amount'],
        input_data['loan_term'],
        input_data['cibil_score'],
        input_data['residential_assets_value'],
        input_data['commercial_assets_value'],
        input_data['luxury_assets_value'],
        input_data['bank_asset_value'],
        prediction,
        approval_prob,
        rejection_prob
    ))
    
    conn.commit()
    conn.close()

def get_user_applications(user_id):
    conn = get_db_connection()
    df = pd.read_sql_query('''
        SELECT * FROM loan_applications 
        WHERE user_id = ? 
        ORDER BY created_at DESC
    ''', conn, params=(user_id,))
    conn.close()
    return df

def get_user_stats(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT 
            COUNT(*) as total_applications,
            SUM(CASE WHEN prediction = 'Approved' THEN 1 ELSE 0 END) as approved,
            SUM(CASE WHEN prediction = 'Rejected' THEN 1 ELSE 0 END) as rejected,
            AVG(approval_probability) as avg_approval_rate
        FROM loan_applications
        WHERE user_id = ?
    ''', (user_id,))
    
    stats = cursor.fetchone()
    conn.close()
    return dict(stats) if stats else None

def log_activity(user_id, activity_type, activity_details):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO user_activity (user_id, activity_type, activity_details)
        VALUES (?, ?, ?)
    ''', (user_id, activity_type, activity_details))
    
    conn.commit()
    conn.close()

init_database()