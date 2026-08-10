# routers/metrics.py — endpoints for chart data
# convergence chart and privacy-utility tradeoff chart

from fastapi import APIRouter
from db import get_db
from schemas import PrivacyResult

router = APIRouter()

@router.get("/convergence")
def get_convergence():
    # returns round-by-round accuracy formatted for the convergence chart
    # React uses this to draw federated vs baseline comparison
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT round_num, accuracy, loss FROM rounds ORDER BY round_num ASC")
    rows = cursor.fetchall()
    conn.close()

    return {
        "federated": [
            {
                "round": row["round_num"],
                "accuracy": row["accuracy"],
                "loss": row["loss"]
            }
            for row in rows
        ]
    }

@router.get("/privacy", response_model=list[PrivacyResult])
def get_privacy_tradeoff():
    # returns the privacy-utility tradeoff results
    # hardcoded for now based on actual experiment results
    # update these values as you run each epsilon level
    return [
        PrivacyResult(epsilon=float('inf'), accuracy=0.5644, training_time_seconds=649.93),
        PrivacyResult(epsilon=10,           accuracy=0.7484, training_time_seconds=15659.90),
        PrivacyResult(epsilon=5,            accuracy=0.0,    training_time_seconds=0.0),
        PrivacyResult(epsilon=1,            accuracy=0.0,    training_time_seconds=0.0),
    ]

@router.get("/summary")
def get_summary():
    # returns aggregated stats across all rounds
    # used for the stats cards at the top of the dashboard
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            COUNT(*) as total_rounds,
            MIN(loss) as best_loss,
            MAX(accuracy) as best_accuracy,
            AVG(accuracy) as avg_accuracy,
            AVG(num_clients) as avg_clients
        FROM rounds
    """)

    row = cursor.fetchone()
    conn.close()

    if row is None:
        return {"message": "no training data yet"}

    return {
        "total_rounds": row["total_rounds"],
        "best_loss": round(row["best_loss"], 4),
        "best_accuracy": round(row["best_accuracy"], 4),
        "avg_accuracy": round(row["avg_accuracy"], 4),
        "avg_clients": round(row["avg_clients"], 1),
    }