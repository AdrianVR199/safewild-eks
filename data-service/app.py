"""
data-service — Persistence microservice for SafeWild classification results.
"""
import json
import logging
import os
import time

import psycopg2
import psycopg2.extras
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("data-service")

app = Flask(__name__)

DB_CONFIG = {
    "host":     os.getenv("POSTGRES_HOST",     "postgres"),
    "port":     int(os.getenv("POSTGRES_PORT", "5432")),
    "dbname":   os.getenv("POSTGRES_DB",       "visionai"),
    "user":     os.getenv("POSTGRES_USER",     "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
}


def get_conn():
    return psycopg2.connect(**DB_CONFIG)


def init_db(retries=10, delay=3):
    for attempt in range(1, retries + 1):
        try:
            conn = get_conn()
            cur  = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS classifications (
                    id               SERIAL PRIMARY KEY,
                    filename         VARCHAR(255),
                    image_base64     TEXT,
                    top_label        VARCHAR(150),
                    common_name      VARCHAR(150),
                    top_confidence   FLOAT,
                    danger           VARCHAR(20),
                    venomous         BOOLEAN DEFAULT FALSE,
                    aggressive       BOOLEAN DEFAULT FALSE,
                    action           TEXT,
                    is_wildlife      BOOLEAN DEFAULT FALSE,
                    all_predictions  JSONB,
                    created_at       TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            conn.commit()
            cur.close()
            conn.close()
            log.info("Database initialised.")
            return
        except psycopg2.OperationalError as exc:
            log.warning("DB not ready (attempt %d/%d): %s", attempt, retries, exc)
            time.sleep(delay)
    raise RuntimeError("Could not connect to PostgreSQL.")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/save", methods=["POST"])
def save():
    d = request.get_json(force=True)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO classifications
                    (filename, image_base64, top_label, common_name, top_confidence,
                     danger, venomous, aggressive, action, is_wildlife, all_predictions)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                RETURNING id
            """, (
                d.get("filename", ""),
                d.get("image_base64", ""),
                d.get("top_label", ""),
                d.get("common_name", ""),
                float(d.get("top_confidence", 0)),
                d.get("danger", "NO_WILDLIFE"),
                bool(d.get("venomous", False)),
                bool(d.get("aggressive", False)),
                d.get("action", ""),
                bool(d.get("is_wildlife", False)),
                json.dumps(d.get("all_predictions", [])),
            ))
            record_id = cur.fetchone()[0]
    return jsonify({"id": record_id, "message": "Saved"}), 201


@app.route("/history", methods=["GET"])
def history():
    page   = max(1, int(request.args.get("page", 1)))
    limit  = min(50, int(request.args.get("limit", 12)))
    offset = (page - 1) * limit

    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT COUNT(*) AS total FROM classifications")
            total = cur.fetchone()["total"]
            cur.execute("""
                SELECT id, filename, image_base64, top_label, common_name,
                       top_confidence, danger, venomous, aggressive, action,
                       is_wildlife, all_predictions,
                       created_at AT TIME ZONE 'UTC' AS created_at
                FROM classifications
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
            """, (limit, offset))
            rows = cur.fetchall()

    items = []
    for r in rows:
        items.append({
            "id":             r["id"],
            "filename":       r["filename"],
            "image_base64":   r["image_base64"],
            "top_label":      r["top_label"],
            "common_name":    r["common_name"],
            "top_confidence": r["top_confidence"],
            "danger":         r["danger"],
            "venomous":       r["venomous"],
            "aggressive":     r["aggressive"],
            "action":         r["action"],
            "is_wildlife":    r["is_wildlife"],
            "all_predictions": r["all_predictions"],
            "created_at":     r["created_at"].isoformat(),
        })
    return jsonify({"items": items, "total": total, "page": page, "limit": limit})


@app.route("/stats", methods=["GET"])
def stats():
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT COUNT(*) AS total FROM classifications")
            total = cur.fetchone()["total"]

            cur.execute("SELECT COUNT(*) AS total FROM classifications WHERE is_wildlife = TRUE")
            total_wildlife = cur.fetchone()["total"]

            cur.execute("SELECT COALESCE(AVG(top_confidence),0) AS avg FROM classifications")
            avg_conf = float(cur.fetchone()["avg"])

            cur.execute("""
                SELECT danger, COUNT(*) AS count
                FROM classifications
                WHERE is_wildlife = TRUE
                GROUP BY danger ORDER BY count DESC
            """)
            by_danger = [dict(r) for r in cur.fetchall()]

            cur.execute("""
                SELECT common_name AS label, COUNT(*) AS count
                FROM classifications
                WHERE is_wildlife = TRUE
                GROUP BY common_name ORDER BY count DESC LIMIT 10
            """)
            top_labels = [dict(r) for r in cur.fetchall()]

            cur.execute("""
                SELECT (created_at AT TIME ZONE 'UTC')::date AS day, COUNT(*) AS count
                FROM classifications
                GROUP BY day ORDER BY day DESC LIMIT 7
            """)
            daily = [{"day": str(r["day"]), "count": r["count"]} for r in cur.fetchall()]

    return jsonify({
        "total":          total,
        "total_wildlife": total_wildlife,
        "avg_confidence": avg_conf,
        "by_danger":      by_danger,
        "top_labels":     top_labels,
        "daily":          daily,
    })


@app.route("/delete/<int:record_id>", methods=["DELETE"])
def delete(record_id):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM classifications WHERE id = %s", (record_id,))
    return jsonify({"message": f"Record {record_id} deleted."})


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5002, debug=False)
