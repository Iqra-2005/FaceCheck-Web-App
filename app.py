import base64
import numpy as np
import cv2
import faiss
import threading, queue
from datetime import date, datetime
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from tensorflow.keras.models import load_model
from werkzeug.security import check_password_hash, generate_password_hash

from db_utils import (
    get_conn ,init_db, init_subjects_table, init_admins_table, init_lectures_table, get_user_by_username, verify_user, get_users_with_embeddings,
    save_attendance, get_attendance_for_user, get_all_users, get_attendance_all, add_user, update_student_info,
    add_subject, get_all_subjects, delete_all_students, get_attendance_by_date, get_attendance_by_rollno, 
    get_student_by_roll, get_students_by_class , set_total_lectures, get_total_lectures, get_all_lectures,
    update_admin_password, add_admin, delete_admin ,  clear_attendance_table, clear_subjects_table
)

# --- Flask initialization---
app = Flask(__name__)
app.secret_key = 'add_a_key_here'

# --- DB Init ---
init_db()
init_subjects_table()
init_lectures_table()
init_admins_table()

# --- Constants ---
EMBEDDING_MODEL_PATH = 'embedding_model.h5'
IMG_SIZE = 105
DIST_THRESHOLD = 0.1 # Updated threshold for recognition

# --- Load embedding model ---
embedding_model = load_model(EMBEDDING_MODEL_PATH)

# --- Embedding helpers ---
def normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x)
    return x if n==0 else x/n

def embed_face_bgr_array(face_bgr: np.ndarray) -> np.ndarray:
    # Convert BGR → RGB
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32)/255.0
    emb = embedding_model.predict(np.expand_dims(face_resized, axis=0), verbose=0)[0]
    return normalize(emb.astype(np.float32))

# --- FAISS Global Cache ---
user_cache = []       # list of user dicts
faiss_index = None     # FAISS index

def build_faiss_index():
    global faiss_index, user_cache

    users = get_users_with_embeddings()
    all_embeddings = []
    user_cache = []

    for u in users:
        embs = u.get("embeddings", [])
        for emb in embs:
            # Ensure embedding is np.array
            emb_arr = np.array(emb, dtype=np.float32)
            all_embeddings.append(emb_arr)
            user_cache.append({
                "id": u["id"],
                "username": u["username"],
                "roll_no": u["roll_no"],
                "full_name": u["full_name"],
                "class_name": u["class_name"]
            })

    if len(all_embeddings) == 0:
        faiss_index = None
        print(" No embeddings found, FAISS index cleared.")
        return

    all_embeddings = np.vstack(all_embeddings).astype("float32")
    d = all_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(all_embeddings)
    faiss_index = index
    print(f"FAISS index built with {len(all_embeddings)} embeddings")

# Build cache at startup
build_faiss_index()



# --- Attendance Queue + Worker ---
attendance_queue = queue.Queue()

def attendance_worker():
    """Background worker to save attendance asynchronously"""
    while True:
        try:
            task = attendance_queue.get()
            if task is None:  # poison pill to exit
                break
            save_attendance(**task)
        except Exception as e:
            print("[Worker] Error saving attendance:", e)
        finally:
            attendance_queue.task_done()

# Start worker thread
threading.Thread(target=attendance_worker, daemon=True).start()

# --- Routes ---

@app.route('/')
def index():
    if session.get('username'):
        return redirect(url_for('dashboard'))
    if session.get('admin'):
        return redirect(url_for('admin_dashboard'))
    return render_template('base.html')



# --- User Guide ----
@app.route('/user_guide')
def user_guide():
    return render_template('user_guide.html')


# ----- User registration/login -----
@app.route('/register', methods=['GET'])
def register():
    return render_template('register.html')


@app.route('/register', methods=['POST'])
def register_submit():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password') or 'default123'
    full_name = data.get('full_name')
    roll_no = data.get('roll_no')
    class_name = data.get('class_name')
    images_b64 = data.get('images', [])

    if not all([username, full_name, roll_no, class_name]) or len(images_b64) < 1:
        return jsonify({"status": "error", "message": "Missing fields or no images submitted."}), 400

    # Check username
    if get_user_by_username(username):
        return jsonify({"status": "error", "message": "Username already exists."}), 400

    embeddings = []
    for img_b64 in images_b64:
        if ',' in img_b64:
            img_b64 = img_b64.split(',')[1]
        try:
            img_bytes = base64.b64decode(img_b64)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                continue
            emb = embed_face_bgr_array(img)
            embeddings.append(emb.tolist())
        except Exception as e:
            print("Error decoding/embedding image:", e)
            continue

    if len(embeddings) == 0:
        return jsonify({"status": "error", "message": "No valid face embeddings extracted."}), 400

    # Check roll number
    existing_student = get_student_by_roll(roll_no)
    if existing_student:
        return jsonify({"status": "error", "message": "Roll number already registered."}), 400

    # Check embeddings similarity with existing users
    users_with_embs = get_users_with_embeddings()
    for user in users_with_embs:
        for e1 in embeddings:
            for e2 in user['embeddings']:
                dist = np.linalg.norm(np.array(e1) - np.array(e2))
                if dist < 0.12:  # similarity threshold (tune as needed)
                    return jsonify({
                        "status": "error",
                        "message": f"Face already registered for user '{user['username']}'"
                    }), 400

    # All checks passed → add user
    add_user(username, password, full_name, roll_no, class_name, embeddings)

    # Rebuild FAISS index asynchronously
    threading.Thread(target=build_faiss_index).start()

    return jsonify({"status": "ok", "message": "Registration successful."})


@app.route('/login', methods=['GET','POST'])
def login():
    if request.method=='GET':
        return render_template('login.html')
    username=request.form.get('username')
    password=request.form.get('password')
    if not username or not password:
        flash("Provide username and password.","danger")
        return redirect(url_for('login'))
    if verify_user(username,password):
        session['username']=username
        return redirect(url_for('dashboard'))
    else:
        flash("Wrong credentials.","danger")
        return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.pop('username',None)
    session.pop('admin',None)
    return redirect(url_for('login'))

# ---- User Dashboard ------

@app.route('/dashboard', methods=['GET'])
def dashboard():
    if not session.get('username'):
        return redirect(url_for('login'))
    
    username = session['username']
    user = get_user_by_username(username)
    if not user:
        return redirect(url_for('login'))

    # Attendance calculations (unchanged)
    attendance = get_attendance_for_user(username)
    subject_counts = {}
    for rec in attendance:
        subject_counts[rec['subject']] = subject_counts.get(rec['subject'], 0) + 1

    total_lectures = get_all_lectures()
    summary = []
    total_attended = 0
    total_possible = 0
    for subject, attended in subject_counts.items():
        total = total_lectures.get(subject, 0)
        percent = (attended / total * 100) if total > 0 else 0
        summary.append({
            'subject': subject,
            'attended': attended,
            'total': total,
            'percent': round(percent, 2)
        })
        total_attended += attended
        total_possible += total

    overall_percent = (total_attended / total_possible * 100) if total_possible > 0 else 0

    subjects = get_all_subjects()

    # Mask password for display
    password_masked = '•' * 8  # never show actual password

    return render_template(
        'dashboard.html',
        username=username,
        full_name=user['full_name'],
        roll_no=user['roll_no'],
        class_name=user['class_name'],
        password_masked=password_masked,
        attendance=attendance,
        subjects=subjects,
        summary=summary,
        overall_percent=round(overall_percent, 2)
    )

@app.route('/student/update_info', methods=['POST'])
def student_update_info():
    # Ensure student is logged in
    if not session.get('username'):
        return jsonify({"success": False, "message": "Unauthorized. Please login."}), 401

    username = session['username']
    user = get_user_by_username(username)
    if not user:
        return jsonify({"success": False, "message": "User not found."}), 404

    data = request.get_json()
    new_roll = data.get('roll_no', '').strip()
    new_class = data.get('class_name', '').strip()
    new_password = data.get('password', '').strip()

    # If nothing to update
    if not any([new_roll, new_class, new_password]):
        return jsonify({"success": False, "message": "No update values provided."})

    conn = get_conn()
    c = conn.cursor()

    try:
        # Check if new roll number is already taken by another user
        if new_roll:
            c.execute("SELECT id FROM users WHERE roll_no=? AND username<>?", (new_roll, username))
            if c.fetchone():
                return jsonify({"success": False, "message": "Roll number already taken by another student."})

        # Build update query dynamically
        updates = []
        params = []

        if new_roll:
            updates.append("roll_no=?")
            params.append(new_roll)
        if new_class:
            updates.append("class_name=?")
            params.append(new_class)
        if new_password:
            updates.append("password_hash=?")
            params.append(generate_password_hash(new_password))

        if updates:
            params.append(username)
            query = f"UPDATE users SET {', '.join(updates)} WHERE username=?"
            c.execute(query, params)
            conn.commit()

        return jsonify({"success": True, "message": "Your information has been updated successfully."})

    except Exception as e:
        conn.rollback()
        return jsonify({"success": False, "message": f"Error updating info: {e}"})
    finally:
        conn.close()


# ----- Admin login -----
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'GET':
        return render_template('admin_login.html')

    email = request.form.get('email')
    password = request.form.get('password')

    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT * FROM admins WHERE email=?", (email,))
    admin = c.fetchone()
    conn.close()

    if admin and check_password_hash(admin["password"], password):
        session['admin'] = True
        session['admin_email'] = admin["email"]
        return redirect(url_for('admin_dashboard'))
    else:
        flash("Invalid email or password.", "danger")
        return redirect(url_for('admin_login'))



@app.route('/admin/logout')
def admin_logout():
    session.pop('admin', None)
    session.pop('admin_email', None)
    return redirect(url_for('index'))


# ----- Admin dashboard -----
@app.route('/admin/dashboard')
def admin_dashboard():
    if not session.get('admin'):
        return redirect(url_for('admin_login'))
    return render_template('admin_dashboard.html')

# ----- Admin - Scan attendance -----



@app.route('/admin/scan', methods=['GET'])
def admin_scan():
    if not session.get('admin'):
        return redirect(url_for('admin_login'))
    subjects = get_all_subjects()
    return render_template('admin_scan.html', subjects=subjects)


@app.route('/admin/scan', methods=['POST'])
def admin_scan_post():
    if not session.get('admin'):
        return jsonify({"status":"error","message":"Not authorized"}), 403


    data = request.get_json()
    subject = data.get('subject','').strip()
    images_b64 = data.get('images',[])

    if not images_b64 or not subject:
        return jsonify({"status":"error","message":"No images or subject"}), 400

    # Ensure subject exists
    add_subject(subject)

    date_str = date.today().isoformat()
    marked = []
    already_marked = []

    for img_b64 in images_b64:
        if ',' in img_b64:
            img_b64 = img_b64.split(',')[1]
        try:
            # Decode image
            img_bytes = base64.b64decode(img_b64)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                continue

            # --- Generate embedding ---
            emb = embed_face_bgr_array(img).reshape(1, -1)

            # Skip if FAISS index is empty
            if faiss_index is None or faiss_index.ntotal == 0:
                continue

            # FAISS search
            D, I = faiss_index.search(emb, 1)
            best_idx = int(I[0][0])
            best_dist = float(D[0][0])

            # Skip if distance too high
            if best_dist > DIST_THRESHOLD:
                continue

            user = user_cache[best_idx]

            # --- Check if already marked today for this subject ---
            existing = get_attendance_by_rollno(user['roll_no'])
            already_for_today = any(a['subject'] == subject and a['date'] == date_str for a in existing)

            student_info = {
                'full_name': user['full_name'],
                'roll_no': user['roll_no'],
                'class_name': user['class_name']
            }

            if already_for_today:
                already_marked.append(student_info)
            else:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                try:
                    save_attendance(
                        user_id=user['id'],
                        username=user['username'],
                        full_name=user['full_name'],
                        roll_no=user['roll_no'],
                        class_name=user['class_name'],
                        subject=subject,
                        date_str=date_str
                    )
                    marked.append(student_info)
                except Exception as e:
                    print("Error saving attendance:", e)
                    continue

        except Exception as e:
            print("Error decoding or processing image:", e)
            continue

    return jsonify({
        "status": "ok",
        "marked": marked,
        "already_marked": already_marked
    })



@app.route('/api/add_subject', methods=['POST'])
def add_subject_api():
    if not session.get('admin'):
        return jsonify({"status":"error","message":"Not authorized"}), 403

    data = request.get_json()
    subject_name = data.get('subject_name', '').strip()
    if not subject_name:
        return jsonify({"status":"error","message":"No subject provided"}), 400

    add_subject(subject_name)  # <-- call your function here
    return jsonify({"status":"ok"})

# ----- Single Face Recognition API -----
@app.route('/recognize_face', methods=['POST'])
def recognize_face():
    data = request.get_json()
    img_b64 = data.get('image')
    if ',' in img_b64:
        img_b64 = img_b64.split(',')[1]

    try:
        img_bytes = base64.b64decode(img_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({
                "username": "Unknown",
                "roll_no": "",
                "full_name": "Unknown",
                "class_name": "",
                "best_dist": None
            })

        emb = embed_face_bgr_array(img).reshape(1, -1)

        if faiss_index is None or faiss_index.ntotal == 0:
            return jsonify({
                "username": "Unknown",
                "roll_no": "",
                "full_name": "Unknown",
                "class_name": "",
                "best_dist": None
            })

        # FAISS search
        D, I = faiss_index.search(emb, 1)
        best_idx = int(I[0][0])
        best_dist = float(D[0][0])

        print(f"[DEBUG] Best match dist={best_dist:.3f}")

        if best_dist > DIST_THRESHOLD:
            return jsonify({
                "username": "Unknown",
                "roll_no": "",
                "full_name": "Unknown",
                "class_name": "",
                "best_dist": best_dist
            })

        user = user_cache[best_idx]

        return jsonify({
            "username": user["username"],
            "roll_no": user["roll_no"],
            "full_name": user["full_name"],
            "class_name": user["class_name"],
            "best_dist": best_dist
        })

    except Exception as e:
        print("Recognition error:", e)
        return jsonify({
            "username": "Unknown",
            "roll_no": "",
            "full_name": "Unknown",
            "class_name": "",
            "best_dist": None
        })

@app.route('/api/already_marked', methods=['POST'])
def already_marked():
    if not session.get('admin'):
        return jsonify({"status":"error","message":"Not authorized"}), 403

    data = request.get_json()
    subject = data.get('subject','').strip()
    if not subject:
        return jsonify({"status":"error","message":"No subject provided"}), 400

    today_str = date.today().isoformat()
    marked_rolls = set()

    # Loop through all users and check their attendance for this subject today
    users = get_users_with_embeddings()  # or get all users
    for user in users:
        existing_attendance = get_attendance_by_rollno(user['roll_no'])
        for a in existing_attendance:
            if a['subject'] == subject and a['date'] == today_str:
                marked_rolls.add(user['roll_no'])
                break

    return jsonify({
        "status":"ok",
        "already_marked": list(marked_rolls)
    })

# ----- Admin - Student management -----
# ----- Admin - Student management -----
@app.route('/admin/students', methods=['GET'])
def admin_students():
    if not session.get('admin'):
        return redirect(url_for('admin_dashboard'))

    # Get filter values from query parameters
    roll_filter = request.args.get('roll_no', '').strip()
    class_filter = request.args.get('class_name', '').strip()

    if roll_filter:
        students = [get_student_by_roll(roll_filter)] if get_student_by_roll(roll_filter) else []
    elif class_filter:
        students = get_students_by_class(class_filter)
    else:
        students = get_all_users()  # all students

    # Sort students by roll_no (assuming it's numeric)
    try:
        students = sorted(students, key=lambda s: int(s['roll_no']))
    except ValueError:
        # fallback in case roll_no is alphanumeric like "21A"
        students = sorted(students, key=lambda s: s['roll_no'])

    # For the class dropdown filter
    classes = sorted(list({s['class_name'] for s in get_all_users()}))

    return render_template(
        'admin_students.html',
        students=students,
        classes=classes,
        selected_roll=roll_filter,
        selected_class=class_filter
    )




@app.route('/admin/delete_class_students', methods=['POST'])
def admin_delete_class_students():
    if 'admin_id' not in session:
        return redirect(url_for('admin_login'))

    class_name = request.form.get('class_name', '').strip()
    if not class_name:
        flash("Please select a class to delete.", "danger")
        return redirect(url_for('admin_dashboard'))

    from db_utils import delete_all_students
    res = delete_all_students(class_name=class_name, confirm=True)
    flash(res['message'], "success" if res['success'] else "warning")
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/delete_student', methods=['POST'])
def admin_delete_student():
    roll = request.form.get('roll_no','').strip()
    if not roll:
        flash("Please provide a roll number.", "danger")
        return redirect(url_for('admin_dashboard'))
    res = delete_all_students(roll_no=roll)  # confirm not required for single delete
    flash(res['message'], "success" if res['success'] else "warning")
    return redirect(url_for('admin_dashboard'))


@app.route('/admin/delete_all_students', methods=['POST'])
def admin_delete_all_students():
    if 'admin_id' not in session:
        return redirect(url_for('admin_login'))
    res = delete_all_students(confirm=True)
    flash(res['message'], "success" if res['success'] else "warning")
    return redirect(url_for('admin_dashboard'))



# ----- Admin - Attendance Records -----
@app.route('/admin/attendance', methods=['GET'])
def admin_attendance_records():
    if not session.get('admin'):
        return redirect(url_for('admin_login'))

    # Get filter values from query parameters (if any)
    date_filter = request.args.get('date')
    roll_filter = request.args.get('roll_no')

    if date_filter:
        attendance = get_attendance_by_date(date_filter)
    elif roll_filter:
        attendance = get_attendance_by_rollno(roll_filter)
    else:
        attendance = get_attendance_all()

    # Collect classes and subjects for dropdown filters
    classes = sorted(list({rec['class_name'] for rec in attendance if rec.get('class_name')}))
    subjects = sorted(list({rec['subject'] for rec in attendance if rec.get('subject')}))

    return render_template(
        'admin_attendance_records.html',
        attendance=attendance,
        classes=classes,
        subjects=subjects,
        selected_date=date_filter,
        selected_roll=roll_filter
    )


from collections import defaultdict

@app.route('/admin/attendance_summary', methods=['GET', 'POST'])
def admin_attendance_summary():
    if not session.get('admin'):
        return redirect(url_for('admin_login'))

    # Get filter values
    class_filter = request.args.get('class_name', '').strip()
    roll_filter = request.args.get('roll_no', '').strip()

    if roll_filter:
        students = [get_student_by_roll(roll_filter)] if get_student_by_roll(roll_filter) else []
    elif class_filter:
        students = get_students_by_class(class_filter)
    else:
        students = get_all_users()

    total_lectures = get_all_lectures()
    summary_data = []

    for student in students:
        attendance = get_attendance_for_user(student['username'])
        subject_counts = {}
        for rec in attendance:
            subject_counts[rec['subject']] = subject_counts.get(rec['subject'], 0) + 1

        subject_summary = []
        total_attended = 0
        total_possible = 0
        for subject, total in total_lectures.items():
            attended = subject_counts.get(subject, 0)
            percent = (attended / total * 100) if total > 0 else 0
            subject_summary.append({
                'subject': subject,
                'attended': attended,
                'total': total,
                'percent': round(percent, 2)
            })
            total_attended += attended
            total_possible += total

        overall_percent = (total_attended / total_possible * 100) if total_possible > 0 else 0

        summary_data.append({
            'student': student['full_name'],
            'roll_no': student['roll_no'],
            'class_name': student['class_name'],
            'summary': subject_summary,
            'overall_percent': round(overall_percent, 2)
        })

    # Group summary_data by class
    summary_by_class = defaultdict(list)
    for s in summary_data:
        summary_by_class[s['class_name']].append(s)

    # Classes for filter dropdown
    classes = sorted(list({s['class_name'] for s in get_all_users()}))

    return render_template(
        'admin_attendance_summary.html',
        summary_data=summary_data,            # available if needed
        summary_by_class=summary_by_class,    # new dict for UI
        total_lectures=total_lectures,
        selected_class=class_filter,
        selected_roll=roll_filter,
        classes=classes,
        subjects=list(total_lectures.keys())  # for dynamic subject headers
    )


@app.route('/admin/set_lectures', methods=['GET','POST'])
def admin_set_lectures():
    if not session.get('admin'):
        return redirect(url_for('admin_login'))

    if request.method == 'POST':
        subject = request.form.get('subject', '').strip()
        total = request.form.get('total', '').strip()
        if subject and total.isdigit():
            set_total_lectures(subject, int(total))
            flash(f"Total lectures for '{subject}' set to {total}.", "success")
        else:
            flash("Please select a subject and enter a valid number.", "danger")
        return redirect(url_for('admin_set_lectures'))

    subjects = get_all_subjects()
    total_lectures = get_all_lectures()
    return render_template('admin_set_lectures.html', subjects=subjects, total_lectures=total_lectures)



# ------------ADMIN SETTIINGS ---------------

@app.route('/admin/settings', methods=['GET', 'POST'])
def admin_settings():
    if not session.get('admin_email'):  # must be logged in
        return redirect(url_for('admin_login'))

    email = session['admin_email']

    if request.method == 'POST':
        action = request.form.get('action')

        # ---------------- CHANGE PASSWORD ----------------
        if action == "change_password":
            current_pw = request.form.get('current_password')
            new_pw = request.form.get('new_password')
            confirm_pw = request.form.get('confirm_password')

            conn = get_conn()
            c = conn.cursor()
            c.execute("SELECT * FROM admins WHERE email=?", (email,))
            admin = c.fetchone()
            conn.close()

            if not admin or not check_password_hash(admin["password"], current_pw):
                flash("Current password is incorrect", "danger")
                return redirect(url_for('admin_settings'))

            if new_pw != confirm_pw:
                flash("New passwords do not match", "danger")
                return redirect(url_for('admin_settings'))

            update_admin_password(email, new_pw)
            flash("Password updated successfully", "success")
            return redirect(url_for('admin_settings'))

        # ---------------- ADD NEW ADMIN ----------------
        elif action == "add_admin":
            new_email = request.form.get('new_admin_email')
            new_password = request.form.get('new_admin_password')

            if not new_email or not new_password:
                flash("Email and password required for new admin", "danger")
                return redirect(url_for('admin_settings'))

            if add_admin(new_email, new_password):
                flash(f"New admin added: {new_email}", "success")
            else:
                flash("Admin with this email already exists", "warning")

            return redirect(url_for('admin_settings'))

        # ---------------- DELETE ADMIN ----------------
        elif action == "delete_admin":
            admin_id = request.form.get('admin_id')

            conn = get_conn()
            c = conn.cursor()
            c.execute("SELECT email FROM admins WHERE id=?", (admin_id,))
            target = c.fetchone()

            if target and target["email"] != email:  # prevent deleting self
                c.execute("DELETE FROM admins WHERE id=?", (admin_id,))
                conn.commit()
                flash("Admin deleted successfully", "success")
            else:
                flash("You cannot delete your own account", "danger")
            conn.close()

            return redirect(url_for('admin_settings'))

    # ---------------- GET ADMINS LIST ----------------
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT id, email FROM admins")
    admins = c.fetchall()
    conn.close()

    return render_template('admin_settings.html', admins=admins)

# ----- Admin Guide Page -----
@app.route('/admin/guide')
def admin_guide():
    if not session.get('admin'):
        return redirect(url_for('admin_login'))
    return render_template('admin_guide.html')


@app.route('/api/clear_attendance', methods=['POST'])
def api_clear_attendance():
    success = clear_attendance_table()
    return jsonify({"status":"ok" if success else "error"})

@app.route('/api/clear_subjects', methods=['POST'])
def api_clear_subjects():
    success = clear_subjects_table()
    return jsonify({"status":"ok" if success else "error"})


# ----- API endpoint -----
@app.route('/api/users')
def api_users():
    if not session.get('admin'):
        return jsonify({"status":"error","message":"unauthenticated"}),403
    users=get_all_users()
    return jsonify({"users":users})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)

