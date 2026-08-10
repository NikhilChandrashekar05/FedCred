# routers/training.py — endpoints for training status and round history
# the React dashboard polls these to show live training progress

from fastapi import APIRouter, HTTPException
from db import get_db
from schemas import RoundMetrics, TrainingStatus

# APIRouter groups related endpoints together
# main.py mounts this under the /training prefix
router = APIRouter()

@router.get("/rounds", response_model=list[RoundMetrics])
def get_rounds():
    # returns all round metrics from the database
    # React uses this to draw the accuracy and loss curves
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT round_num, loss, accuracy, num_clients FROM rounds ORDER BY round_num ASC")
    rows = cursor.fetchall()
    conn.close()

    # convert each sqlite Row object to a dictionary
    return [dict(row) for row in rows]

@router.get("/status", response_model=TrainingStatus)
def get_status():
    # returns current training status
    # checks how many rounds are in the database vs total expected
    conn = get_db()
    cursor = conn.cursor()

    # get the latest round number and its accuracy
    cursor.execute("SELECT round_num, accuracy FROM rounds ORDER BY round_num DESC LIMIT 1")
    latest = cursor.fetchone()
    conn.close()

    # if no rounds exist yet training hasn't started
    if latest is None:
        return TrainingStatus(
            status="idle",
            current_round=0,
            total_rounds=10,
            latest_accuracy=0.0
        )

    current_round = latest['round_num']
    latest_accuracy = latest['accuracy']

    # if all 10 rounds are done training is complete
    status = "complete" if current_round >= 10 else "training"

    return TrainingStatus(
        status=status,
        current_round=current_round,
        total_rounds=10,
        latest_accuracy=latest_accuracy
    )