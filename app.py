
from flask import Flask, render_template, request, redirect, session
from flask_sqlalchemy import SQLAlchemy
from text_evaluation import verify_text
from datetime import datetime
import json

app = Flask(__name__)

# Session configuration
app.secret_key = "halluchek_secret_key"

# Use your existing SQLite database: instance/users.db
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


# =========================
# USER TABLE (EXISTING)
# =========================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)


# =========================
# SCAN HISTORY TABLE (NEW)
# =========================
class ScanHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id"),
        nullable=False,
        index=True
    )

    input_text = db.Column(db.Text, nullable=False)
    total = db.Column(db.Integer, default=0)
    supported = db.Column(db.Integer, default=0)
    suspected = db.Column(db.Integer, default=0)
    unverified = db.Column(db.Integer, default=0)
    results_json = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    @property
    def results(self):
        try:
            return json.loads(self.results_json)
        except (json.JSONDecodeError, TypeError):
            return []


# =========================
# HOME PAGE
# =========================
@app.route("/")
def home():
    return render_template("index.html")


# =========================
# REGISTER
# =========================
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"].strip()
        email = request.form["email"].strip().lower()
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]

        if not name or not email or not password:
            return "Please fill in all required fields."

        if password != confirm_password:
            return "Passwords do not match."

        existing_user = User.query.filter_by(email=email).first()

        if existing_user:
            return "Email already registered."

        new_user = User(
            name=name,
            email=email,
            password=password
        )

        try:
            db.session.add(new_user)
            db.session.commit()
        except Exception:
            db.session.rollback()
            app.logger.exception("Registration failed")
            return "Registration failed. Please try again."

        return redirect("/login")

    return render_template("register.html")


# =========================
# LOGIN
# =========================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        password = request.form["password"]

        user = User.query.filter_by(email=email).first()

        if user and user.password == password:
            session.clear()
            session["user_id"] = user.id
            session["user_name"] = user.name
            return redirect("/dashboard")

        return "Invalid email or password."

    return render_template("login.html")


# =========================
# DASHBOARD
# =========================
@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect("/login")

    return render_template(
        "dashboard.html",
        username=session.get("user_name"),
        submitted_text="",
        report=None,
        error=None
    )


# =========================
# VERIFY TEXT AND SAVE SCAN
# =========================
@app.route("/verify", methods=["POST"])
def verify():
    if "user_id" not in session:
        return redirect("/login")

    text = request.form.get("textInput", "").strip()

    if not text:
        return render_template(
            "dashboard.html",
            username=session.get("user_name"),
            submitted_text=text,
            report=None,
            error="Please enter some text to verify."
        )

    try:
        # Run your existing verification system
        report = verify_text(text)

        # Check the expected report structure
        required_keys = [
            "results", "total", "supported",
            "suspected", "unverified"
        ]

        if not isinstance(report, dict) or not all(
            key in report for key in required_keys
        ):
            raise ValueError(
                "verify_text() returned an unexpected report format."
            )

        # Save this scan in the existing database
        scan = ScanHistory(
            user_id=session["user_id"],
            input_text=text,
            total=report["total"],
            supported=report["supported"],
            suspected=report["suspected"],
            unverified=report["unverified"],
            results_json=json.dumps(report["results"])
        )

        db.session.add(scan)
        db.session.commit()

        return render_template(
            "dashboard.html",
            username=session.get("user_name"),
            submitted_text=text,
            report=report,
            error=None
        )

    except Exception:
        db.session.rollback()
        app.logger.exception("Text verification or scan saving failed")

        return render_template(
            "dashboard.html",
            username=session.get("user_name"),
            submitted_text=text,
            report=None,
            error=(
                "Verification could not be completed. "
                "Please check the application terminal for details."
            )
        )


# =========================
# HISTORY - CURRENT USER ONLY
# =========================
@app.route("/history")
def history():
    if "user_id" not in session:
        return redirect("/login")

    scans = (
        ScanHistory.query
        .filter_by(user_id=session["user_id"])
        .order_by(ScanHistory.created_at.desc())
        .all()
    )

    return render_template(
        "history.html",
        username=session.get("user_name"),
        scans=scans
    )


# =========================
# ABOUT
# =========================
@app.route("/about")
def about():
    return render_template("about.html")


# =========================
# LOGOUT
# =========================
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


# =========================
# CREATE MISSING TABLES
# =========================
with app.app_context():
    db.create_all()


# =========================
# RUN APPLICATION
# =========================
if __name__ == "__main__":
    app.run(debug=True)
