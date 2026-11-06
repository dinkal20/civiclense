import os
import re
import io
import base64
import binascii
import logging

from datetime import datetime, timezone, timedelta
from functools import wraps

from flask import Flask, request, jsonify, session, send_from_directory
from flask_cors import CORS
from pymongo.errors import DuplicateKeyError
from bson import ObjectId
from bson.errors import InvalidId
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image

from config import (
    client,
    db,
    users_collection,
    admin_collection,
    reports_collection,
    departments_collection,
    test_database_connection,
)


# ==================================================
# APP CONFIGURATION
# ==================================================

app = Flask(__name__)

secret_key = os.getenv("SECRET_KEY", "").strip()

if not secret_key:
    raise RuntimeError(
        "SECRET_KEY is missing. Add a strong SECRET_KEY to backend/.env."
    )

app.config.update(
    SECRET_KEY=secret_key,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=True,
    PERMANENT_SESSION_LIFETIME=timedelta(hours=8),
)

CORS(app, supports_credentials=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_IMAGE_SIZE = 8 * 1024 * 1024

# AI review threshold
AI_CONFIDENCE_THRESHOLD = 0.40
AI_LOW_CONFIDENCE_REVIEW = 0.60

# Hosted vision API is the only image classifier.
# No local YOLO, PyTorch or OpenCV model is loaded.
try:
    import vision_fallback
except Exception:
    vision_fallback = None
    logger.exception("Hosted vision API module could not be loaded")


# ==================================================
# COMMON HELPERS
# ==================================================

def utc_now():
    return datetime.now(timezone.utc)


def valid_email(email):
    return bool(
        isinstance(email, str)
        and re.fullmatch(
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
            email.strip(),
        )
    )




def current_account():
    return session.get("account")


def login_required(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        if not current_account():
            return jsonify({"error": "Please log in first."}), 401
        return function(*args, **kwargs)

    return wrapper


def role_required(*allowed_roles):
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            account = current_account()

            if not account:
                return jsonify({"error": "Please log in first."}), 401

            if account.get("role") not in allowed_roles:
                return jsonify({"error": "You are not authorized."}), 403

            return function(*args, **kwargs)

        return wrapper

    return decorator


def iso_date(value):
    return value.isoformat() if isinstance(value, datetime) else value


def report_to_dict(report, include_email=False):
    report_id = str(report.get("_id", ""))
    stored_image = report.get("image")
    image_url = None

    # Preserve the original uploaded image MIME type.
    # Older reports without this field default to JPEG.
    image_mime_type = report.get("image_mime_type") or "image/jpeg"

    if isinstance(stored_image, str) and stored_image.strip():
        if stored_image.startswith("data:image/"):
            image_url = stored_image
        else:
            image_url = (
                f"data:{image_mime_type};base64,{stored_image}"
            )

    manual_reassignment = report.get("manual_reassignment")
    if not isinstance(manual_reassignment, dict):
        manual_reassignment = None
    else:
        manual_reassignment = dict(manual_reassignment)
        for key in ("requested_at", "assigned_at"):
            if key in manual_reassignment:
                manual_reassignment[key] = iso_date(
                    manual_reassignment[key]
                )

    rejection_request = report.get("rejection_request")
    if isinstance(rejection_request, dict):
        rejection_request = dict(rejection_request)
        for key in ("requested_at", "reviewed_at"):
            if key in rejection_request:
                rejection_request[key] = iso_date(rejection_request[key])

    created_at = iso_date(report.get("created_at"))

    # A report can carry several AI categories.
    primary_category = report.get("ai_category")
    ai_categories = report.get("ai_categories")

    if not isinstance(ai_categories, list):
        ai_categories = []

    ai_categories = [
        item.strip() for item in ai_categories
        if isinstance(item, str) and item.strip()
    ]

    if (
        isinstance(primary_category, str)
        and primary_category.strip()
        and primary_category.strip() not in ai_categories
    ):
        ai_categories.insert(0, primary_category.strip())

    category_display = ", ".join(ai_categories) if ai_categories else None

    result = {
        "id": report_id,
        "report_id": report_id,
        "image": image_url,
        "image_url": image_url,
        "image_mime_type": image_mime_type,
        "latitude": report.get("latitude"),
        "longitude": report.get("longitude"),
        "address": report.get("address", ""),
        "location": report.get("address", ""),
        "location_method": report.get("location_method", "manual"),
        "description": report.get("description", ""),
        "status": report.get("status", "Pending"),
        "ai_category": category_display,
        "ai_categories": ai_categories,
        "ai_primary_category": primary_category,
        "ai_detections": report.get("ai_detections", []),
        "ai_engine": report.get("ai_engine", "vision"),
        "ai_vision": report.get("ai_vision"),
        "issue_type": category_display or "Unclassified",
        "predicted_class": category_display,
        "confidence": report.get("confidence"),
        "ai_review_status": report.get("ai_review_status"),
        "created_at": created_at,
        "date": created_at,
        "department": report.get("department"),
        "severity": report.get("severity"),
        "ai_severity": report.get("severity"),
        "rejection_reason": report.get("rejection_reason", ""),
        "rejection_request": rejection_request,
        "manual_reassignment": manual_reassignment,
    }

    if include_email:
        result["email"] = report.get("email")

    return result


def get_model_categories():
    """Categories supported by CivicLens hosted image classification."""
    return [
        "pothole",
        "garbage",
        "broken_streetlight",
        "water_leakage",
    ]


def get_allocated_categories(exclude_department_id=None):
    query = {"categories": {"$exists": True, "$ne": []}}

    if exclude_department_id is not None:
        query["_id"] = {"$ne": exclude_department_id}

    allocated = set()

    for department in departments_collection.find(query, {"categories": 1}):
        categories = department.get("categories", [])

        if isinstance(categories, list):
            allocated.update(
                item.strip().casefold()
                for item in categories
                if isinstance(item, str) and item.strip()
            )

    return allocated


def get_unassigned_categories():
    allocated = get_allocated_categories()

    return [
        category
        for category in get_model_categories()
        if category.casefold() not in allocated
    ]


def validate_unassigned_categories(categories):
    if not isinstance(categories, list) or not categories:
        return None, "Select at least one available AI category."

    all_categories = get_model_categories()
    allocated = get_allocated_categories()
    selected = []
    selected_keys = set()

    for submitted in categories:
        if not isinstance(submitted, str):
            return None, "Invalid category selection."

        submitted = submitted.strip()

        matching = next(
            (
                category for category in all_categories
                if category.casefold() == submitted.casefold()
            ),
            None,
        )

        if matching is None:
            return None, f"Invalid AI category: {submitted}"

        key = matching.casefold()

        if key in allocated:
            return None, (
                f'"{matching}" is already assigned to another department. '
                "Refresh the page and choose an available category."
            )

        if key not in selected_keys:
            selected_keys.add(key)
            selected.append(matching)

    return selected, None


# ==================================================
# HOSTED VISION CLASSIFICATION
# ==================================================

def classify_with_second_opinion(image, image_bytes, mime_type="image/jpeg"):
    """Classify a civic issue using only the hosted vision API."""

    categories = get_model_categories()

    analysis = {
        "categories": [],
        "category": None,
        "confidence": 0.0,
        "detections": [],
        "needs_review": True,
        "engine": "vision",
        "vision": None,
        "yolo_claim": [],
        "vision_agrees": False,
        "unverified": False,
        "review_reason": None,
    }

    if vision_fallback is None or not vision_fallback.is_enabled():
        analysis["unverified"] = True
        analysis["review_reason"] = (
            "Hosted image classification is unavailable. "
            "Admin review is required."
        )
        return analysis

    verdict = vision_fallback.classify(
        image_bytes,
        categories,
        mime_type=mime_type,
    )

    analysis["vision"] = verdict

    if verdict.get("status") == "category":
        category = verdict.get("category")

        try:
            confidence = float(verdict.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        # Accept only categories supported by CivicLens.
        matching_category = next(
            (
                item for item in categories
                if isinstance(category, str)
                and item.casefold() == category.strip().casefold()
            ),
            None,
        )

        if matching_category is None:
            analysis["review_reason"] = (
                "Hosted AI returned an unsupported category. "
                "Admin review is required."
            )
            return analysis

        category = matching_category
        confidence = max(0.0, min(confidence, 1.0))

        analysis["categories"] = [category]
        analysis["category"] = category
        analysis["confidence"] = confidence
        analysis["detections"] = [{
            "category": category,
            "confidence": confidence,
            "detections": 1,
        }]
        analysis["needs_review"] = False
        analysis["review_reason"] = (
            "Hosted AI classification requires admin verification."
        )

        logger.info(
            "Hosted vision classified report as %s (%.2f)",
            category,
            confidence,
        )

    elif verdict.get("status") == "none":
        analysis["review_reason"] = (
            "The hosted AI could not identify a visible civic issue."
        )

    else:
        analysis["unverified"] = True
        analysis["review_reason"] = (
            "Hosted AI was unavailable. Admin review is required."
        )

        logger.warning(
            "Hosted vision unavailable: %s",
            verdict.get("reason"),
        )

    return analysis


# ==================================================
# DEPARTMENT HELPERS
# ==================================================

def find_departments_for_categories(categories):
    """Map each detected category to its active department.

    Returns (routed, unrouted) where routed is a list of
    (category, department) pairs, de-duplicated by department.
    """
    routed = []
    unrouted = []
    seen = set()

    for category in categories:
        department = find_department_for_category(category)

        if department is None:
            unrouted.append(category)
            continue

        key = str(department.get("_id"))

        if key in seen:
            continue

        seen.add(key)
        routed.append((category, department))

    return routed, unrouted


def find_department_for_category(category):
    if not isinstance(category, str) or not category.strip():
        return None

    category_key = category.strip().casefold()

    for department in departments_collection.find({
        "status": "active",
        "categories": {"$exists": True},
    }):
        categories = department.get("categories", [])

        if isinstance(categories, list) and any(
            isinstance(item, str)
            and item.strip().casefold() == category_key
            for item in categories
        ):
            return department

    return None


def serialize_department(department):
    return {
        "id": str(department.get("_id", "")),
        "name": department.get("name", ""),
        "email": department.get("email", ""),
        "categories": department.get("categories", []),
        "status": department.get("status", "active"),
        "created_at": iso_date(department.get("created_at")),
    }


def get_object_id(value):
    try:
        return ObjectId(value)
    except (InvalidId, TypeError):
        return None


def find_active_department(data):
    """Find an active department by ID or name."""
    department_id = str(data.get("department_id", "")).strip()
    department_name = str(
        data.get("department_name", data.get("department", ""))
    ).strip()

    if department_id:
        object_id = get_object_id(department_id)

        if object_id is None:
            return None

        return departments_collection.find_one({
            "_id": object_id,
            "status": "active",
        })

    if not department_name:
        return None

    return departments_collection.find_one({
        "name": {
            "$regex": "^" + re.escape(department_name) + "$",
            "$options": "i",
        },
        "status": "active",
    })


# ==================================================
# HEALTH CHECK AND PAGE ROUTES
# ==================================================

@app.route("/", methods=["GET"])
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/css/<path:filename>")
def serve_css(filename):
    return send_from_directory(os.path.join(BASE_DIR, "css"), filename)


@app.route("/js/<path:filename>")
def serve_js(filename):
    return send_from_directory(os.path.join(BASE_DIR, "js"), filename)


@app.route("/login.html")
def login_page():
    return send_from_directory(BASE_DIR, "login.html")


@app.route("/register.html")
def register_page():
    return send_from_directory(BASE_DIR, "register.html")


@app.route("/report.html")
def report_page():
    return send_from_directory(BASE_DIR, "report.html")


@app.route("/my-reports.html")
def my_reports_page():
    return send_from_directory(BASE_DIR, "my-reports.html")


@app.route("/department.html")
def department_page():
    return send_from_directory(BASE_DIR, "department.html")


@app.route("/admin.html")
def admin_page():
    return send_from_directory(BASE_DIR, "admin.html")


@app.route("/api/health", methods=["GET"])
def health():
    try:
        test_database_connection()
        return jsonify({"status": "ok", "database": "connected"}), 200
    except Exception:
        logger.exception("Database health check failed")
        return jsonify({
            "status": "error",
            "database": "disconnected",
        }), 503


# ==================================================
# CITIZEN REGISTRATION
# ==================================================

@app.route("/api/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}

    name = str(data.get("name", "")).strip()
    email = str(data.get("email", "")).strip().lower()
    password = data.get("password", "")

    if not name or not email or not password:
        return jsonify({
            "error": "Name, email and password are required."
        }), 400

    if not valid_email(email):
        return jsonify({"error": "Enter a valid email address."}), 400

    if not isinstance(password, str) or len(password) < 8:
        return jsonify({
            "error": "Password must contain at least 8 characters."
        }), 400

    if (
        users_collection.find_one({"email": email})
        or admin_collection.find_one({"email": email})
        or departments_collection.find_one({"email": email})
    ):
        return jsonify({
            "error": "An account with this email already exists."
        }), 409

    try:
        users_collection.insert_one({
            "name": name,
            "email": email,
            "password_hash": generate_password_hash(password),
            "created_at": utc_now(),
        })
    except DuplicateKeyError:
        return jsonify({
            "error": "This email is already registered."
        }), 409
    except Exception:
        logger.exception("Citizen registration failed")
        return jsonify({
            "error": "Registration failed. Please try again."
        }), 500

    return jsonify({"message": "Registration successful."}), 201


# ==================================================
# LOGIN: CITIZEN, DEPARTMENT AND ADMIN
# ==================================================

@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}

    email = str(data.get("email", "")).strip().lower()
    password = data.get("password", "")

    if not email or not isinstance(password, str) or not password:
        return jsonify({
            "error": "Email and password are required."
        }), 400

    # 1. Check admin collection
    admin = admin_collection.find_one({"email": email})

    if admin and check_password_hash(
        admin.get("password_hash", ""),
        password,
    ):
        account = {
            "name": admin.get("name", ""),
            "email": admin.get("email", email),
            "role": "admin",
        }

        session.clear()
        session["account"] = account
        session.permanent = True

        return jsonify({
            "message": "Login successful.",
            "user": account,
        }), 200

    # 2. Check citizen collection
    user = users_collection.find_one({"email": email})

    if user and check_password_hash(
        user.get("password_hash", ""),
        password,
    ):
        account = {
            "name": user.get("name", ""),
            "email": user.get("email", email),
            "role": "user",
        }

        session.clear()
        session["account"] = account
        session.permanent = True

        return jsonify({
            "message": "Login successful.",
            "user": account,
        }), 200

    # 3. Check department collection
    department = departments_collection.find_one({"email": email})

    if department and check_password_hash(
        department.get("password_hash", ""),
        password,
    ):
        if str(department.get("status", "active")).lower() != "active":
            return jsonify({
                "error": "This department account is inactive."
            }), 403

        account = {
            "name": department.get("name", ""),
            "email": department.get("email", email),
            "role": "department",
            "department": department.get("name", ""),
            "categories": department.get("categories", []),
        }

        session.clear()
        session["account"] = account
        session.permanent = True

        return jsonify({
            "message": "Login successful.",
            "user": account,
        }), 200

    return jsonify({
        "error": "Invalid email or password."
    }), 401

@app.route("/api/me", methods=["GET"])
@login_required
def get_current_user():
    return jsonify({"user": current_account()}), 200


@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"message": "Logged out successfully."}), 200


# ==================================================
# ADMIN: AVAILABLE AI CATEGORIES
# ==================================================

@app.route("/api/admin/categories", methods=["GET"])
@login_required
@role_required("admin")
def admin_get_categories():
    return jsonify({
        "categories": get_unassigned_categories(),
        "all_categories": get_model_categories(),
    }), 200


# ==================================================
# CREATE REPORT
# ==================================================

@app.route("/api/reports", methods=["POST"])
@login_required
@role_required("user")
def create_report():
    data = request.get_json(silent=True) or {}
    account = current_account()

    description = str(data.get("description", "")).strip()
    image_data = data.get("image")

    if not description:
        return jsonify({"error": "Description is required."}), 400

    if not isinstance(image_data, str) or not image_data:
        return jsonify({"error": "A report image is required."}), 400

    mime_type = "image/jpeg"

    if image_data.startswith("data:") and "," in image_data:
        header, image_data = image_data.split(",", 1)

        if header.startswith("data:") and ";" in header:
            mime_type = header[5:].split(";", 1)[0].strip().lower()

    try:
        image_bytes = base64.b64decode(image_data, validate=True)
    except (binascii.Error, ValueError):
        return jsonify({"error": "The image format is invalid."}), 400

    if not image_bytes:
        return jsonify({"error": "The image is empty."}), 400

    if len(image_bytes) > MAX_IMAGE_SIZE:
        return jsonify({
            "error": "Image must be 8 MB or smaller."
        }), 413

    try:
        with Image.open(io.BytesIO(image_bytes)) as uploaded_image:
            detected_mime_type = Image.MIME.get(uploaded_image.format)

            if detected_mime_type:
                mime_type = detected_mime_type

            uploaded_image.verify()

    except Exception:
        return jsonify({"error": "Please upload a valid image."}), 400

    if not mime_type.startswith("image/"):
        return jsonify({"error": "Unsupported image format."}), 400

    try:
        analysis = classify_with_second_opinion(
            None,
            image_bytes,
            mime_type,
        )
    except Exception:
        logger.exception("Hosted image classification failed")
        return jsonify({
            "error": "Image analysis failed. Please try again."
        }), 500

    confidence = analysis["confidence"]

    # Missing, weak or ambiguous prediction needs manual admin review.
    is_unclassified = analysis["needs_review"]

    categories = [] if is_unclassified else analysis["categories"]
    category = None if is_unclassified else analysis["category"]

    routed, unrouted = find_departments_for_categories(categories)

    department = routed[0][1] if routed else None
    department_name = department.get("name") if department else None
    department_names = [item[1].get("name") for item in routed]

    # One photo can show more than one issue.
    multi_department = len(department_names) > 1
    extra_issues = department_names[1:] + unrouted

    # Lower-confidence classifications are flagged for human review.
    low_confidence = (
        not is_unclassified
        and confidence < AI_LOW_CONFIDENCE_REVIEW
    )

    # Every hosted AI classification is reviewed by a person.
    from_vision = (
        analysis.get("engine") == "vision"
        and not analysis.get("unverified")
    )

    unverified = bool(analysis.get("unverified"))

    # Nothing to route -> admin has to assign it manually.
    needs_manual_assignment = is_unclassified or department is None

    # Routed, but a person should look at it.
    needs_additional_review = (
        not needs_manual_assignment
        and (
            multi_department
            or bool(unrouted)
            or low_confidence
            or from_vision
            or unverified
        )
    )

    manual_reassignment = None

    if needs_manual_assignment:
        reason = (
            (
                analysis.get("review_reason")
                or "AI could not confidently classify this report."
            )
            if is_unclassified
            else "No active department is assigned to the AI category."
        )

        manual_reassignment = {
            "status": "pending",
            "reason": reason,
            "requested_by": "AI System",
            "requested_at": utc_now(),
            "previous_department": "",
            "request_type": "manual_assignment",
        }

    elif needs_additional_review:
        notes = []

        if multi_department or unrouted:
            notes.append(
                "Report shows more than one issue ("
                + ", ".join(categories)
                + ")."
            )

        if multi_department:
            notes.append(
                "It also concerns "
                + ", ".join(department_names[1:])
                + "."
            )

        if unrouted:
            notes.append(
                "No active department owns "
                + ", ".join(unrouted)
                + "."
            )

        if from_vision:
            verdict = analysis.get("vision") or {}

            detail = (
                ": " + verdict["reason"]
                if verdict.get("reason")
                else ""
            )

            notes.append(
                "Hosted vision AI classified this as \"%s\" "
                "(%.0f%% confidence%s) - please confirm."
                % (category, confidence * 100, detail)
            )

        elif unverified:
            notes.append(
                "Hosted vision AI was unavailable. "
                "The report needs manual verification."
            )

        elif low_confidence:
            notes.append(
                "AI is only %.0f%% sure of \"%s\" - please confirm."
                % (confidence * 100, category)
            )

        notes.append("Assigned to " + str(department_name) + " meanwhile.")

        manual_reassignment = {
            "status": "pending",
            "reason": " ".join(notes),
            "requested_by": "AI System",
            "requested_at": utc_now(),
            "previous_department": department_name,
            "request_type": "additional_department",
        }

    latitude = data.get("latitude", data.get("lat"))
    longitude = data.get("longitude", data.get("lng", data.get("long")))

    try:
        latitude = float(latitude) if latitude not in (None, "") else None
        longitude = float(longitude) if longitude not in (None, "") else None
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid location coordinates."}), 400

    if latitude is not None and not -90 <= latitude <= 90:
        return jsonify({"error": "Latitude is out of range."}), 400

    if longitude is not None and not -180 <= longitude <= 180:
        return jsonify({"error": "Longitude is out of range."}), 400

    location_method = str(data.get("location_method", "manual")).lower()

    if location_method not in ("live", "manual"):
        location_method = "manual"

    report = {
        "email": account["email"].lower(),
        "description": description,
        "image": base64.b64encode(image_bytes).decode("utf-8"),
        "image_mime_type": mime_type,
        "latitude": latitude,
        "longitude": longitude,
        "address": str(data.get("address", "")).strip(),
        "location_method": location_method,
        "ai_category": category,
        "ai_categories": categories,
        "ai_detections": analysis["detections"],
        "ai_departments": department_names,
        "ai_unrouted_categories": unrouted,
        "ai_extra_issue_departments": extra_issues,
        "ai_engine": analysis.get("engine", "vision"),
        "ai_vision": analysis.get("vision"),
        # Kept for compatibility with existing report data and frontend.
        "ai_yolo_claim": analysis.get("yolo_claim"),
        "ai_unverified": unverified,
        "confidence": confidence,
        "ai_review_status": (
            "pending" if needs_manual_assignment else "classified"
        ),
        "department": department_name,
        "status": "Pending",
        "manual_reassignment": manual_reassignment,
        "created_at": utc_now(),
    }

    try:
        result = reports_collection.insert_one(report)
    except Exception:
        logger.exception("Saving report failed")
        return jsonify({"error": "Could not save your report."}), 500

    return jsonify({
        "message": "Report submitted successfully.",
        "report_id": str(result.inserted_id),
        "ai_category": report["ai_category"],
        "ai_categories": categories,
        "ai_engine": analysis.get("engine", "vision"),
        "confidence": confidence,
        "department": department_name,
        "status": "Pending",
        "ai_review_status": report["ai_review_status"],
        "manual_assignment_required": needs_manual_assignment,
        "additional_review_required": needs_additional_review,
    }), 201


# ==================================================
# CITIZEN: VIEW OWN REPORTS
# ==================================================

@app.route("/api/reports/<email>", methods=["GET"])
@login_required
def get_user_reports(email):
    account = current_account()
    email = email.strip().lower()

    # Citizens may only view their own reports.
    # Departments must use /api/department/reports.
    if account.get("role") == "user":
        if account.get("email", "").lower() != email:
            return jsonify({"error": "You are not authorized."}), 403
    elif account.get("role") != "admin":
        return jsonify({"error": "You are not authorized."}), 403

    reports = list(
        reports_collection.find({"email": email}).sort("created_at", -1)
    )

    return jsonify({
        "reports": [report_to_dict(report) for report in reports]
    }), 200


# ==================================================
# ADMIN: VIEW ALL REPORTS
# ==================================================

@app.route("/api/admin/reports", methods=["GET"])
@login_required
@role_required("admin")
def admin_get_reports():
    reports = list(reports_collection.find().sort("created_at", -1))

    return jsonify({
        "reports": [
            report_to_dict(report, include_email=True)
            for report in reports
        ]
    }), 200


# ==================================================
# ADMIN: DIRECTLY REJECT A REPORT
# ==================================================

@app.route("/api/admin/reports/<report_id>/reject", methods=["POST"])
@login_required
@role_required("admin")
def admin_reject_report(report_id):
    object_id = get_object_id(report_id)

    if object_id is None:
        return jsonify({"error": "Invalid report ID."}), 400

    data = request.get_json(silent=True) or {}
    reason = str(data.get("reason", "")).strip()

    if len(reason) < 5:
        return jsonify({
            "error": "Please enter a rejection reason of at least 5 characters."
        }), 400

    report = reports_collection.find_one({"_id": object_id})

    if not report:
        return jsonify({"error": "Report not found."}), 404

    if str(report.get("status", "")).strip().lower() == "rejected":
        return jsonify({
            "error": "This report has already been rejected."
        }), 409

    account = current_account()
    now = utc_now()

    update_fields = {
        "status": "Rejected",
        "rejection_reason": reason,
        "admin_rejection": {
            "reason": reason,
            "rejected_by": account.get("email", ""),
            "rejected_at": now,
        },
    }

    existing_request = report.get("rejection_request")

    if (
        isinstance(existing_request, dict)
        and str(existing_request.get("status", "")).lower() == "pending"
    ):
        update_fields.update({
            "rejection_request.status": "disapproved",
            "rejection_request.reviewed_at": now,
            "rejection_request.reviewed_by": account.get("email", ""),
            "rejection_request.admin_note":
                "Report rejected directly by admin.",
        })

    result = reports_collection.update_one(
        {"_id": object_id, "status": report.get("status")},
        {"$set": update_fields},
    )

    if result.modified_count != 1:
        return jsonify({
            "error": "The report changed. Refresh and try again."
        }), 409

    return jsonify({
        "message": "Report rejected successfully.",
        "status": "Rejected",
        "rejection_reason": reason,
        "report_id": str(object_id),
    }), 200


# ==================================================
# ADMIN: REVIEW DEPARTMENT REJECTION REQUESTS
# ==================================================

@app.route("/api/admin/rejection-requests", methods=["GET"])
@login_required
@role_required("admin")
def admin_get_rejection_requests():
    reports = list(reports_collection.find({
        "status": {
            "$regex": "^Rejection Requested$",
            "$options": "i",
        },
        "rejection_request.status": {
            "$regex": "^pending$",
            "$options": "i",
        },
    }).sort("rejection_request.requested_at", -1))

    requests_list = []

    for report in reports:
        rejection_request = report.get("rejection_request")

        if not isinstance(rejection_request, dict):
            continue

        requests_list.append({
            **report_to_dict(report, include_email=True),
            "complaint_id": str(report.get("_id", "")),
            "department_name": (
                rejection_request.get("requested_by")
                or report.get("department")
                or "Unassigned"
            ),
            "reason": rejection_request.get("reason", ""),
            "requested_at": iso_date(
                rejection_request.get("requested_at")
            ),
            "rejection_status": "pending",
            "reviewed_at": iso_date(rejection_request.get("reviewed_at")),
            "reviewed_by": rejection_request.get("reviewed_by", ""),
            "admin_note": rejection_request.get("admin_note", ""),
        })

    return jsonify({"requests": requests_list}), 200


def review_rejection_request(report_id, decision):
    object_id = get_object_id(report_id)

    if object_id is None:
        return jsonify({"error": "Invalid report ID."}), 400

    report = reports_collection.find_one({"_id": object_id})

    if not report:
        return jsonify({"error": "Report not found."}), 404

    rejection_request = report.get("rejection_request")

    if not isinstance(rejection_request, dict):
        return jsonify({
            "error": "No rejection request exists for this report."
        }), 409

    if str(rejection_request.get("status", "")).lower() != "pending":
        return jsonify({
            "error": "This rejection request has already been reviewed."
        }), 409

    if str(report.get("status", "")).strip().lower() != "rejection requested":
        return jsonify({
            "error": "The report is no longer awaiting rejection review."
        }), 409

    account = current_account()
    now = utc_now()

    if decision == "approved":
        new_status = "Rejected"
        message = "Rejection request approved."
        rejection_reason = rejection_request.get("reason", "")
    else:
        new_status = "Pending"
        message = "Rejection request disapproved. The report is pending again."
        rejection_reason = ""

    result = reports_collection.update_one(
        {
            "_id": object_id,
            "status": report.get("status"),
            "rejection_request.status": "pending",
        },
        {"$set": {
            "status": new_status,
            "rejection_reason": rejection_reason,
            "rejection_request.status": decision,
            "rejection_request.reviewed_at": now,
            "rejection_request.reviewed_by": account.get("email", ""),
        }},
    )

    if result.modified_count != 1:
        return jsonify({
            "error": "The request changed while you were reviewing it. Refresh and try again."
        }), 409

    return jsonify({
        "message": message,
        "status": new_status,
        "rejection_status": decision,
        "report_id": str(object_id),
    }), 200


@app.route(
    "/api/admin/rejection-requests/<report_id>/approve",
    methods=["POST"],
)
@login_required
@role_required("admin")
def admin_approve_rejection_request(report_id):
    return review_rejection_request(report_id, "approved")


@app.route(
    "/api/admin/rejection-requests/<report_id>/disapprove",
    methods=["POST"],
)
@login_required
@role_required("admin")
def admin_disapprove_rejection_request(report_id):
    return review_rejection_request(report_id, "disapproved")


# ==================================================
# ADMIN: DEPARTMENT MANAGEMENT
# ==================================================

@app.route("/api/admin/departments", methods=["GET"])
@login_required
@role_required("admin")
def admin_get_departments():
    departments = list(
        departments_collection.find().sort("created_at", -1)
    )

    return jsonify({
        "departments": [
            serialize_department(department)
            for department in departments
        ]
    }), 200


@app.route("/api/admin/departments", methods=["POST"])
@login_required
@role_required("admin")
def admin_create_department():
    data = request.get_json(silent=True) or {}

    name = str(data.get("name", "")).strip()
    email = str(data.get("email", "")).strip().lower()
    password = data.get("password", "")
    status = str(data.get("status", "active")).strip().lower()
    categories = data.get("categories", [])

    if not name or not email or not password:
        return jsonify({
            "error": "Department name, email and password are required."
        }), 400

    if not valid_email(email):
        return jsonify({"error": "Enter a valid email address."}), 400

    if not isinstance(password, str) or len(password) < 8:
        return jsonify({
            "error": "Password must contain at least 8 characters."
        }), 400

    if status not in ("active", "inactive"):
        return jsonify({
            "error": "Status must be active or inactive."
        }), 400

   if (
        users_collection.find_one({"email": email})
        or admin_collection.find_one({"email": email})
        or departments_collection.find_one({"email": email})
    ):
        return jsonify({
            "error": "An account with this email already exists."
        }), 409

    selected_categories, category_error = (
        validate_unassigned_categories(categories)
    )

    if category_error:
        status_code = 409 if "already assigned" in category_error else 400
        return jsonify({"error": category_error}), status_code

    department = {
        "name": name,
        "email": email,
        "password_hash": generate_password_hash(password),
        "categories": selected_categories,
        "status": status,
        "created_at": utc_now(),
    }

    try:
        # Recheck category allocation immediately before insertion.
        selected_categories, category_error = (
            validate_unassigned_categories(selected_categories)
        )

        if category_error:
            status_code = (
                409 if "already assigned" in category_error else 400
            )
            return jsonify({"error": category_error}), status_code

        department["categories"] = selected_categories
        result = departments_collection.insert_one(department)

    except DuplicateKeyError:
        return jsonify({
            "error": "An account with this email already exists."
        }), 409
    except Exception:
        logger.exception("Department creation failed")
        return jsonify({
            "error": "Could not create the department account."
        }), 500

    return jsonify({
        "message": "Department account created successfully.",
        "department": {
            "id": str(result.inserted_id),
            "name": name,
            "email": email,
            "categories": selected_categories,
            "status": status,
        },
    }), 201


@app.route(
    "/api/admin/departments/<department_id>/status",
    methods=["PATCH"],
)
@login_required
@role_required("admin")
def admin_update_department_status(department_id):
    object_id = get_object_id(department_id)

    if object_id is None:
        return jsonify({"error": "Invalid department ID."}), 400

    data = request.get_json(silent=True) or {}
    status = str(data.get("status", "")).strip().lower()

    if status not in ("active", "inactive"):
        return jsonify({
            "error": "Status must be active or inactive."
        }), 400

    result = departments_collection.update_one(
        {"_id": object_id},
        {"$set": {"status": status}},
    )

    if result.matched_count == 0:
        return jsonify({"error": "Department not found."}), 404

    return jsonify({
        "message": f"Department account is now {status}."
    }), 200


# ==================================================
# ADMIN: MANUAL ASSIGNMENT / REASSIGNMENT QUEUE
# ==================================================

@app.route("/api/admin/reassignment-requests", methods=["GET"])
@login_required
@role_required("admin")
def admin_get_reassignment_requests():
    reports = list(reports_collection.find({
        "$or": [
            {
                "manual_reassignment.status": {
                    "$regex": "^pending$",
                    "$options": "i",
                }
            },
            {
                "ai_category": None,
                "manual_reassignment": {"$exists": False},
            },
            {
                "confidence": {"$lt": AI_CONFIDENCE_THRESHOLD},
                "manual_reassignment": {"$exists": False},
            },
        ]
    }).sort("created_at", -1))

    requests_list = []

    for report in reports:
        reassignment = report.get("manual_reassignment")

        # Add a pending assignment request for older unclassified reports.
        if not isinstance(reassignment, dict):
            category = report.get("ai_category")
            confidence = report.get("confidence", 0) or 0

            if (
                category is not None
                and confidence >= AI_CONFIDENCE_THRESHOLD
            ):
                continue

            reassignment = {
                "status": "pending",
                "reason": "AI could not confidently classify this report.",
                "requested_by": "AI System",
                "requested_at": report.get("created_at"),
                "previous_department": "",
                "request_type": "manual_assignment",
            }

            reports_collection.update_one(
                {
                    "_id": report["_id"],
                    "manual_reassignment": {"$exists": False},
                },
                {"$set": {"manual_reassignment": reassignment}},
            )

        if str(reassignment.get("status", "")).lower() != "pending":
            continue

        requested_by = reassignment.get("requested_by", "")
        request_type = reassignment.get("request_type")

        if request_type not in (
            "manual_assignment", "reassignment", "additional_department",
        ):
            request_type = (
                "manual_assignment"
                if str(requested_by).strip().casefold() == "ai system"
                else "reassignment"
            )

        requests_list.append({
            **report_to_dict(report, include_email=True),
            "complaint_id": str(report["_id"]),
            "reason": reassignment.get("reason", ""),
            "requested_by": requested_by or "AI System",
            "requested_at": iso_date(
                reassignment.get("requested_at")
            ),
            "previous_department": reassignment.get(
                "previous_department", ""
            ),
            "reassignment_status": "pending",
            "request_type": request_type,
            "request_label": (
                "Manual Report Assignment"
                if request_type == "manual_assignment"
                else "Additional Issue Detected"
                if request_type == "additional_department"
                else "Department Reassignment"
            ),
            "ai_review_status": report.get(
                "ai_review_status", "pending"
            ),
        })

    return jsonify({"requests": requests_list}), 200


# ==================================================
# ADMIN: ASSIGN OR REASSIGN A REPORT
# ==================================================

@app.route(
    "/api/admin/reports/<report_id>/assign-department",
    methods=["POST"],
)
@login_required
@role_required("admin")
def admin_assign_reassignment_department(report_id):
    object_id = get_object_id(report_id)

    if object_id is None:
        return jsonify({"error": "Invalid report ID."}), 400

    data = request.get_json(silent=True) or {}
    report = reports_collection.find_one({"_id": object_id})

    if not report:
        return jsonify({"error": "Report not found."}), 404

    reassignment = report.get("manual_reassignment")

    if not isinstance(reassignment, dict):
        return jsonify({
            "error": "This report is not awaiting manual assignment."
        }), 409

    if str(reassignment.get("status", "")).lower() != "pending":
        return jsonify({
            "error": "This report is not awaiting assignment."
        }), 409

    department = find_active_department(data)

    if not department:
        return jsonify({
            "error": "The selected department was not found or is inactive."
        }), 404

    actual_department_name = department.get("name", "").strip()

    if not actual_department_name:
        return jsonify({"error": "The department name is invalid."}), 400

    account = current_account()
    now = utc_now()

    result = reports_collection.update_one(
        {
            "_id": object_id,
            "manual_reassignment.status": "pending",
        },
        {"$set": {
            "department": actual_department_name,
            "status": "Pending",
            "ai_review_status": "manually_assigned",
            "manual_reassignment.status": "assigned",
            "manual_reassignment.assigned_department":
                actual_department_name,
            "manual_reassignment.assigned_department_id":
                str(department["_id"]),
            "manual_reassignment.assigned_by":
                account.get("email", ""),
            "manual_reassignment.assigned_at": now,
        }},
    )

    if result.modified_count != 1:
        return jsonify({
            "error": "The request changed. Refresh and try again."
        }), 409

    return jsonify({
        "message": "Report assigned successfully.",
        "department": actual_department_name,
        "department_id": str(department["_id"]),
        "status": "Pending",
        "ai_review_status": "manually_assigned",
        "reassignment_status": "assigned",
    }), 200


# ==================================================
# DEPARTMENT: ACCESS A REPORT
# ==================================================

def get_department_owned_report(report_id):
    object_id = get_object_id(report_id)

    if object_id is None:
        return None, None, (
            jsonify({"error": "Invalid report ID."}), 400
        )

    account = current_account()
    department_email = str(
        account.get("email", "")
    ).strip().lower()

    department = departments_collection.find_one({
        "email": department_email
    })

    if not department:
        return None, None, (
            jsonify({"error": "Department account not found."}), 404
        )

    report = reports_collection.find_one({"_id": object_id})

    if not report:
        return None, department, (
            jsonify({"error": "Report not found."}), 404
        )

    reassignment = report.get("manual_reassignment")

    if (
        isinstance(reassignment, dict)
        and str(reassignment.get("status", "")).lower() == "pending"
        and reassignment.get("request_type") != "additional_department"
    ):
        return None, department, (
            jsonify({
                "error": "This report is awaiting admin assignment."
            }), 409
        )

    department_name = str(department.get("name", "")).strip()
    assigned_department = report.get("department")

    if assigned_department:
        if (
            str(assigned_department).strip().casefold()
            != department_name.casefold()
        ):
            return None, department, (
                jsonify({
                    "error": "This report is assigned to another department."
                }), 403
            )
    else:
        # Legacy reports without a department can use category matching.
        categories = department.get("categories", [])
        category = report.get("ai_category")

        if (
            isinstance(reassignment, dict)
            or not isinstance(categories, list)
            or not isinstance(category, str)
            or category.casefold() not in [
                item.strip().casefold()
                for item in categories
                if isinstance(item, str)
            ]
        ):
            return None, department, (
                jsonify({
                    "error": "This report is not assigned to your department."
                }), 403
            )

    return report, department, None


# ==================================================
# DEPARTMENT: REQUEST REASSIGNMENT
# ==================================================

@app.route(
    "/api/department/reports/<report_id>/wrong-department",
    methods=["POST"],
)
@login_required
@role_required("department")
def department_request_reassignment(report_id):
    report, department, error_response = get_department_owned_report(
        report_id
    )

    if error_response:
        return error_response

    data = request.get_json(silent=True) or {}
    reason = str(data.get("reason", "")).strip()

    if len(reason) < 5:
        return jsonify({
            "error": "Please provide a reason of at least 5 characters."
        }), 400

    if str(report.get("status", "Pending")).strip().lower() != "pending":
        return jsonify({
            "error": "Only pending reports can be sent for reassignment."
        }), 409

    existing_request = report.get("manual_reassignment")

    if (
        isinstance(existing_request, dict)
        and str(existing_request.get("status", "")).lower() == "pending"
    ):
        return jsonify({
            "error": "This report is already awaiting reassignment."
        }), 409

    rejection_request = report.get("rejection_request")

    if (
        isinstance(rejection_request, dict)
        and str(rejection_request.get("status", "")).lower() == "pending"
    ):
        return jsonify({
            "error": "A rejection request is already awaiting admin review."
        }), 409

    account = current_account()
    department_name = department.get("name", "")

    reassignment = {
        "status": "pending",
        "reason": reason,
        "requested_by": department_name,
        "requested_by_email": account.get("email", ""),
        "requested_at": utc_now(),
        "previous_department": department_name,
        "request_type": "reassignment",
    }

    result = reports_collection.update_one(
        {
            "_id": report["_id"],
            "status": report.get("status", "Pending"),
            "$or": [
                {"manual_reassignment.status": {"$ne": "pending"}},
                {"manual_reassignment.request_type":
                    "additional_department"},
            ],
        },
        {"$set": {
            "department": None,
            "status": "Reassignment Requested",
            "manual_reassignment": reassignment,
        }},
    )

    if result.modified_count != 1:
        return jsonify({
            "error": "The report changed. Refresh and try again."
        }), 409

    return jsonify({
        "message": "Reassignment request sent to admin.",
        "status": "Reassignment Requested",
    }), 200


# ==================================================
# DEPARTMENT: REQUEST REPORT REJECTION
# ==================================================

@app.route(
    "/api/department/reports/<report_id>/reject-request",
    methods=["POST"],
)
@login_required
@role_required("department")
def department_request_report_rejection(report_id):
    report, department, error_response = get_department_owned_report(
        report_id
    )

    if error_response:
        return error_response

    data = request.get_json(silent=True) or {}
    reason = str(data.get("reason", "")).strip()

    if len(reason) < 5:
        return jsonify({
            "error": "Please provide a reason of at least 5 characters."
        }), 400

    status = str(report.get("status", "Pending")).strip().lower()

    if status != "pending":
        return jsonify({
            "error": "Only pending reports can be submitted for rejection."
        }), 409

    existing_request = report.get("rejection_request")

    if (
        isinstance(existing_request, dict)
        and str(existing_request.get("status", "")).lower() == "pending"
    ):
        return jsonify({
            "error": "A rejection request is already awaiting admin review."
        }), 409

    rejection_request = {
        "status": "pending",
        "reason": reason,
        "requested_by": department.get("name", ""),
        "requested_by_email": department.get("email", ""),
        "requested_at": utc_now(),
    }

    result = reports_collection.update_one(
        {
            "_id": report["_id"],
            "status": report.get("status", "Pending"),
        },
        {"$set": {
            "status": "Rejection Requested",
            "rejection_reason": reason,
            "rejection_request": rejection_request,
        }},
    )

    if result.modified_count != 1:
        return jsonify({
            "error": "The report changed. Refresh and try again."
        }), 409

    return jsonify({
        "message": "Rejection request sent to admin for review.",
        "status": "Rejection Requested",
    }), 200


# ==================================================
# DEPARTMENT: ACCEPT REPORT
# ==================================================

@app.route(
    "/api/department/reports/<report_id>/accept",
    methods=["POST"],
)
@login_required
@role_required("department")
def department_accept_report(report_id):
    report, department, error_response = get_department_owned_report(
        report_id
    )

    if error_response:
        return error_response

    if str(report.get("status", "Pending")).strip().lower() != "pending":
        return jsonify({
            "error": "Only pending reports can be accepted."
        }), 409

    result = reports_collection.update_one(
        {
            "_id": report["_id"],
            "status": report.get("status", "Pending"),
            "department": department.get("name"),
            "$or": [
                {"manual_reassignment.status": {"$ne": "pending"}},
                {"manual_reassignment.request_type":
                    "additional_department"},
            ],
        },
        {
            "$set": {
                "status": "In Progress",
                "accepted_at": utc_now(),
                "accepted_by_department": department.get("name", ""),
            }
        },
    )

    if result.modified_count != 1:
        return jsonify({
            "error": "The report changed. Refresh and try again."
        }), 409

    return jsonify({
        "message": "Report accepted successfully.",
        "status": "In Progress",
    }), 200


# ==================================================
# DEPARTMENT: MARK REPORT AS COMPLETED
# ==================================================

@app.route(
    "/api/department/reports/<report_id>/complete",
    methods=["POST"],
)
@login_required
@role_required("department")
def department_complete_report(report_id):
    report, department, error_response = get_department_owned_report(
        report_id
    )

    if error_response:
        return error_response

    status = str(report.get("status", "Pending")).strip().lower()

    if status not in ("in progress", "in_progress"):
        return jsonify({
            "error": "Only reports in progress can be marked completed."
        }), 409

    result = reports_collection.update_one(
        {
            "_id": report["_id"],
            "status": report.get("status"),
        },
        {"$set": {
            "status": "Completed",
            "completed_at": utc_now(),
            "completed_by_department": department.get("name", ""),
        }},
    )

    if result.modified_count != 1:
        return jsonify({
            "error": "The report changed. Refresh and try again."
        }), 409

    return jsonify({
        "message": "Report marked as completed successfully.",
        "status": "Completed",
    }), 200


# ==================================================
# DEPARTMENT: VIEW ASSIGNED REPORTS
# ==================================================

@app.route("/api/department/reports", methods=["GET"])
@login_required
@role_required("department")
def department_get_reports():
    account = current_account()
    department_email = str(
        account.get("email", "")
    ).strip().lower()

    department = departments_collection.find_one({
        "email": department_email
    })

    if not department:
        return jsonify({"error": "Department account not found."}), 404

    department_name = str(department.get("name", "")).strip()
    categories = department.get("categories", [])
    conditions = []

    if department_name:
        conditions.append({
            "department": {
                "$regex": "^" + re.escape(department_name) + "$",
                "$options": "i",
            }
        })

    # Legacy records only: don't expose pending manual assignments or
    # reassignment requests through category-based fallback.
    valid_categories = [
        item.strip()
        for item in categories
        if isinstance(item, str) and item.strip()
    ] if isinstance(categories, list) else []

    if valid_categories:
        conditions.append({
            "department": {"$in": [None, ""]},
            "$or": [
                {"ai_category": {"$in": valid_categories}},
                {"ai_categories": {"$in": valid_categories}},
            ],
            "manual_reassignment": {"$exists": False},
        })

    if not conditions:
        return jsonify({"reports": []}), 200

    # A pending additional-issue note does not block the assigned
    # department from working on the report.
    reports = list(reports_collection.find({
        "$and": [
            {"$or": conditions},
            {"$or": [
                {"manual_reassignment.status": {"$ne": "pending"}},
                {"manual_reassignment.request_type":
                    "additional_department"},
            ]},
        ],
    }).sort("created_at", -1))

    return jsonify({
        "reports": [
            report_to_dict(report, include_email=True)
            for report in reports
        ]
    }), 200


# ==================================================
# START SERVER
# ==================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
    )
