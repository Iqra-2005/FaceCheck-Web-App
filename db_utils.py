import sqlite3
import json
from werkzeug.security import generate_password_hash, check_password_hash
from typing import List, Optional, Dict

DB = 'face_attendance.db'

def get_conn():
    conn = sqlite3.connect(DB, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

# ---------------- INITIALIZATION ----------------
def init_db():
    conn = get_conn()
    c = conn.cursor()
    # Users table
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password_hash TEXT,
        full_name TEXT,
        roll_no TEXT,
        class_name TEXT,
        embeddings_json TEXT
    )
    ''')
    # Attendance table
    c.execute('''
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        username TEXT,
        full_name TEXT,
        roll_no TEXT,
        class_name TEXT,
        subject TEXT,
        date TEXT,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    ''')
    conn.commit()
    conn.close()

def init_subjects_table():
    conn = get_conn()
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS subjects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE
    )
    ''')
    conn.commit()
    conn.close()

def init_lectures_table():
    conn = get_conn()
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS lectures (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject TEXT UNIQUE,
        total_lectures INTEGER
    )
    ''')
    conn.commit()
    conn.close()

def init_admins_table():
    conn = get_conn()
    c = conn.cursor()
    # create table if not exists
    c.execute('''
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()

    # check if table is empty
    c.execute("SELECT COUNT(*) as cnt FROM admins")
    row = c.fetchone()
    if row["cnt"] == 0:
        # insert default admin if no admin exists
        default_email = "admin@gmail.com"
        default_pw = "admin123"
        hashed_pw = generate_password_hash(default_pw)
        c.execute("INSERT INTO admins (email, password) VALUES (?, ?)", (default_email, hashed_pw))
        conn.commit()
        print(f"[INIT] Default admin created → {default_email} / {default_pw}")

    conn.close()


# ---------------- SUBJECT FUNCTIONS ----------------
def add_subject(subject_name: str):
    if not subject_name: 
        return
    conn = get_conn()
    c = conn.cursor()
    try:
        c.execute('INSERT OR IGNORE INTO subjects (name) VALUES (?)', (subject_name,))
        conn.commit()
    except Exception as e:
        print("Error adding subject:", e)
    finally:
        conn.close()

def get_all_subjects() -> List[str]:
    conn = get_conn()
    c = conn.cursor()
    c.execute('SELECT name FROM subjects ORDER BY name ASC')
    rows = c.fetchall()
    conn.close()
    return [r['name'] for r in rows]

# ---------------- USER FUNCTIONS ----------------

def get_user_by_username(username: str) -> Optional[Dict]:
    conn = get_conn()
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username=?', (username,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    user = dict(row)
    user['embeddings'] = json.loads(user['embeddings_json']) if user['embeddings_json'] else []
    return user

def verify_user(username: str, password: str) -> bool:
    conn = get_conn()
    c = conn.cursor()
    c.execute('SELECT password_hash FROM users WHERE username=?', (username,))
    row = c.fetchone()
    conn.close()
    if not row:
        return False
    return check_password_hash(row['password_hash'], password)

def get_all_users() -> List[Dict]:
    conn = get_conn()
    c = conn.cursor()
    c.execute('SELECT id, username, full_name, roll_no, class_name FROM users ORDER BY roll_no ASC')
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_users_with_embeddings() -> List[Dict]:
    conn = get_conn()
    c = conn.cursor()
    c.execute('SELECT * FROM users')
    rows = c.fetchall()
    conn.close()
    users = []
    for r in rows:
        d = dict(r)
        d['embeddings'] = json.loads(d['embeddings_json']) if d['embeddings_json'] else []
        users.append(d)
    return users

'''def delete_all_students():
    conn = get_conn()
    c = conn.cursor()
    try:
        c.execute('DELETE FROM attendance')  # delete all attendance records first
        c.execute('DELETE FROM users')       # delete all users
        conn.commit()
    except Exception as e:
        print("Error deleting all students:", e)
    finally:
        conn.close()'''

def delete_all_students(roll_no: str = None, class_name: str = None, confirm: bool = False) -> dict:
    """
    Flexible delete function:
      - If roll_no is provided: deletes that single user and their attendance.
      - Else if class_name is provided: deletes all users in that class (requires confirm=True).
      - Else: deletes ALL users and attendance (requires confirm=True).

    Returns a dict:
      {
        "success": bool,
        "message": str,
        "users_deleted": int,
        "attendance_deleted": int
      }
    """
    conn = get_conn()
    c = conn.cursor()
    try:
        # --- delete by roll_no ---
        if roll_no:
            roll_no = roll_no.strip()
            print(f"[DELETE] Requested delete by roll_no='{roll_no}'")
            c.execute("SELECT id FROM users WHERE roll_no=?", (roll_no,))
            row = c.fetchone()
            if not row:
                msg = f"No student found with roll_no='{roll_no}'."
                print("[DELETE]", msg)
                return {"success": False, "message": msg, "users_deleted": 0, "attendance_deleted": 0}

            user_id = row["id"]
            c.execute("SELECT COUNT(*) as cnt FROM attendance WHERE user_id=?", (user_id,))
            att_cnt = c.fetchone()["cnt"]
            c.execute("DELETE FROM attendance WHERE user_id=?", (user_id,))
            c.execute("DELETE FROM users WHERE id=?", (user_id,))
            conn.commit()
            msg = f"Deleted student with roll_no='{roll_no}'."
            print(f"[DELETE] {msg} (attendance_deleted={att_cnt})")
            return {"success": True, "message": msg, "users_deleted": 1, "attendance_deleted": att_cnt}

        # --- delete by class_name ---
        elif class_name:
            class_name = class_name.strip()
            print(f"[DELETE] Requested delete by class_name='{class_name}'")
            c.execute("SELECT COUNT(*) as cnt FROM users WHERE class_name=?", (class_name,))
            users_cnt = c.fetchone()["cnt"]
            if users_cnt == 0:
                msg = f"No students found in class '{class_name}'."
                print("[DELETE]", msg)
                return {"success": False, "message": msg, "users_deleted": 0, "attendance_deleted": 0}

            if not confirm:
                msg = f"This will delete {users_cnt} students from class '{class_name}'. Call with confirm=True to proceed."
                print("[DELETE] Confirmation required:", msg)
                return {"success": False, "message": msg, "users_deleted": 0, "attendance_deleted": 0}

            c.execute("SELECT COUNT(*) as cnt FROM attendance WHERE class_name=?", (class_name,))
            att_cnt = c.fetchone()["cnt"]
            c.execute("DELETE FROM attendance WHERE class_name=?", (class_name,))
            c.execute("DELETE FROM users WHERE class_name=?", (class_name,))
            conn.commit()
            msg = f"Deleted {users_cnt} students from class '{class_name}'."
            print(f"[DELETE] {msg} (attendance_deleted={att_cnt})")
            return {"success": True, "message": msg, "users_deleted": users_cnt, "attendance_deleted": att_cnt}

        # --- delete all ---
        else:
            print("[DELETE] Requested delete ALL students")
            c.execute("SELECT COUNT(*) as cnt FROM users")
            users_cnt = c.fetchone()["cnt"]
            if users_cnt == 0:
                msg = "No users found (nothing to delete)."
                print("[DELETE]", msg)
                return {"success": False, "message": msg, "users_deleted": 0, "attendance_deleted": 0}

            if not confirm:
                msg = f"This will delete ALL users ({users_cnt}). Call with confirm=True to proceed."
                print("[DELETE] Confirmation required:", msg)
                return {"success": False, "message": msg, "users_deleted": 0, "attendance_deleted": 0}

            c.execute("SELECT COUNT(*) as cnt FROM attendance")
            att_cnt = c.fetchone()["cnt"]
            c.execute("DELETE FROM attendance")
            c.execute("DELETE FROM users")
            conn.commit()
            msg = f"Deleted ALL users ({users_cnt}) and {att_cnt} attendance records."
            print(f"[DELETE] {msg}")
            return {"success": True, "message": msg, "users_deleted": users_cnt, "attendance_deleted": att_cnt}

    except Exception as e:
        print("[DELETE] Error:", e)
        return {"success": False, "message": f"Error deleting students: {e}", "users_deleted": 0, "attendance_deleted": 0}
    finally:
        conn.close()


def update_student_info(username: str, new_roll_no: str = None, new_class_name: str = None, new_password: str = None) -> dict:
    """
    Update roll number, class, and/or password for a student.
    
    Parameters:
    - username (str): the student's username (cannot be changed)
    - new_roll_no (str, optional): new roll number
    - new_class_name (str, optional): new class
    - new_password (str, optional): new password
    
    Returns:
    - dict: {
        "success": bool,
        "message": str
      }
    """
    if not any([new_roll_no, new_class_name, new_password]):
        return {"success": False, "message": "No new data provided to update."}

    conn = get_conn()
    c = conn.cursor()

    try:
        # Check if user exists
        c.execute("SELECT * FROM users WHERE username=?", (username,))
        user = c.fetchone()
        if not user:
            return {"success": False, "message": f"User '{username}' not found."}

        # Prepare update fields
        updates = []
        values = []

        if new_roll_no:
            # Ensure roll_no is unique
            c.execute("SELECT id FROM users WHERE roll_no=? AND username!=?", (new_roll_no, username))
            if c.fetchone():
                return {"success": False, "message": f"Roll number '{new_roll_no}' already exists."}
            updates.append("roll_no=?")
            values.append(new_roll_no)

        if new_class_name:
            updates.append("class_name=?")
            values.append(new_class_name)

        if new_password:
            password_hash = generate_password_hash(new_password)
            updates.append("password_hash=?")
            values.append(password_hash)

        # Nothing to update?
        if not updates:
            return {"success": False, "message": "No valid updates provided."}

        # Build SQL
        sql = f"UPDATE users SET {', '.join(updates)} WHERE username=?"
        values.append(username)

        c.execute(sql, tuple(values))
        conn.commit()
        return {"success": True, "message": "Student information updated successfully."}

    except Exception as e:
        return {"success": False, "message": f"Error updating student info: {e}"}
    finally:
        conn.close()

# ---------------- ATTENDANCE FUNCTIONS ----------------
def save_attendance(user_id, username, full_name, roll_no, class_name, subject, date_str):
    conn = get_conn()
    c = conn.cursor()
    c.execute('''
    INSERT INTO attendance (user_id, username, full_name, roll_no, class_name, subject, date)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, username, full_name, roll_no, class_name, subject, date_str))
    conn.commit()
    conn.close()



def get_attendance_for_user(username: str) -> List[Dict]:
    conn = get_conn()
    c = conn.cursor()
    c.execute('SELECT date, subject, roll_no, class_name FROM attendance WHERE username=? ORDER BY date DESC', (username,))
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_attendance_all() -> List[Dict]:
    conn = get_conn()
    c = conn.cursor()
    c.execute('SELECT * FROM attendance ORDER BY date DESC')
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]

# ---------------- ATTENDANCE FILTERS ----------------
def get_attendance_by_date(date_str: str) -> List[Dict]:
    """Fetch attendance records for a specific date"""
    conn = get_conn()
    c = conn.cursor()
    c.execute('SELECT * FROM attendance WHERE date=? ORDER BY class_name, roll_no ASC', (date_str,))
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_attendance_by_rollno(roll_no: str) -> List[Dict]:
    """Fetch attendance records for a specific student using roll number"""
    conn = get_conn()
    c = conn.cursor()
    c.execute('SELECT * FROM attendance WHERE roll_no=? ORDER BY date DESC', (roll_no,))
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_student_by_roll(roll_no: str) -> Optional[Dict]:
    """
    Fetch a single student using roll number.
    Returns None if not found.
    """
    conn = get_conn()
    c = conn.cursor()
    c.execute('SELECT id, username, full_name, roll_no, class_name FROM users WHERE roll_no=?', (roll_no,))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None

def get_students_by_class(class_name: str) -> List[Dict]:
    """
    Fetch all students in a specific class.
    Returns an empty list if none found.
    """
    conn = get_conn()
    c = conn.cursor()
    c.execute('SELECT id, username, full_name, roll_no, class_name FROM users WHERE class_name=? ORDER BY roll_no ASC', (class_name,))
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def set_total_lectures(subject: str, total: int):
    conn = get_conn()
    c = conn.cursor()
    c.execute('INSERT OR REPLACE INTO lectures (subject, total_lectures) VALUES (?, ?)', (subject, total))
    conn.commit()
    conn.close()

def get_total_lectures(subject: str) -> int:
    conn = get_conn()
    c = conn.cursor()
    c.execute('SELECT total_lectures FROM lectures WHERE subject=?', (subject,))
    row = c.fetchone()
    conn.close()
    return row['total_lectures'] if row else 0

def get_all_lectures() -> Dict[str, int]:
    conn = get_conn()
    c = conn.cursor()
    c.execute('SELECT subject, total_lectures FROM lectures')
    rows = c.fetchall()
    conn.close()
    return {r['subject']: r['total_lectures'] for r in rows}


def add_user(username: str, password: str, full_name: str, roll_no: str, class_name: str, embeddings: List[List[float]]):
    conn = get_conn()
    c = conn.cursor()
    password_hash = generate_password_hash(password)
    embeddings_json = json.dumps(embeddings)
    c.execute('''
    INSERT INTO users (username, password_hash, full_name, roll_no, class_name, embeddings_json)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (username, password_hash, full_name, roll_no, class_name, embeddings_json))
    conn.commit()
    conn.close()


def update_admin_password(email: str, new_password: str):
    conn = get_conn()
    c = conn.cursor()
    hashed_pw = generate_password_hash(new_password)
    c.execute("UPDATE admins SET password=? WHERE email=?", (hashed_pw, email))
    conn.commit()
    conn.close()

# ---------------- ADD ADMIN ----------------
def add_admin(email: str, password: str) -> bool:
    conn = get_conn()
    c = conn.cursor()

    # check if admin exists
    c.execute("SELECT * FROM admins WHERE email=?", (email,))
    if c.fetchone():
        conn.close()
        return False

    hashed_pw = generate_password_hash(password)
    c.execute("INSERT INTO admins (email, password) VALUES (?, ?)", (email, hashed_pw))
    conn.commit()
    conn.close()
    return True

# ---------------- DELETE ADMIN ----------------
def delete_admin(admin_id: int) -> bool:
    conn = get_conn()
    c = conn.cursor()

    # check if admin exists
    c.execute("SELECT * FROM admins WHERE id=?", (admin_id,))
    if not c.fetchone():
        conn.close()
        return False

    c.execute("DELETE FROM admins WHERE id=?", (admin_id,))
    conn.commit()
    conn.close()
    return True




def clear_attendance_table():
    """
    Deletes all records from the attendance table.
    """
    conn = get_conn()
    c = conn.cursor()
    try:
        c.execute("DELETE FROM attendance")
        conn.commit()
        print("Attendance table cleared successfully.")
        return True
    except Exception as e:
        print("Error clearing attendance table:", e)
        return False
    finally:
        conn.close()


def clear_subjects_table():
    """
    Deletes all records from the subjects table.
    """
    conn = get_conn()
    c = conn.cursor()
    try:
        c.execute("DELETE FROM subjects")
        conn.commit()
        print("Subjects table cleared successfully.")
        return True
    except Exception as e:
        print("Error clearing subjects table:", e)
        return False
    finally:
        conn.close()

