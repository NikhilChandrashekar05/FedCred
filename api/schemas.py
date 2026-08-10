# schemas.py — Pydantic models for all request and response types
# FastAPI uses these to validate incoming data and serialize outgoing data
# if the data doesn't match the schema FastAPI automatically returns a 422 error

from pydantic import BaseModel
from typing import Optional

class RoundMetrics(BaseModel):
    # represents one round of federated training from the database
    round_num: int
    loss: float
    accuracy: float
    num_clients: int

class TrainingStatus(BaseModel):
    # current state of the federated training system
    status: str          # "idle", "training", or "complete"
    current_round: int
    total_rounds: int
    latest_accuracy: float

class LoanApplication(BaseModel):
    # input from the loan application form
    # these match exactly the 16 features the model was trained on
    loan_amnt: float
    int_rate: float
    installment: float
    annual_inc: float
    dti: float
    fico_range_low: float
    fico_range_high: float
    inq_last_6mths: float
    open_acc: float
    pub_rec: float
    revol_bal: float
    revol_util: float
    total_acc: float
    delinq_2yrs: float
    emp_length: float
    home_ownership: float

class LoanDecision(BaseModel):
    # response from the predict endpoint
    decision: str           # "APPROVED" or "DENIED"
    probability: float      # model's confidence score 0-1
    reasons: list[str]      # top 3 SHAP reason codes

class PrivacyResult(BaseModel):
    # one data point for the privacy-utility tradeoff chart
    epsilon: float
    accuracy: float
    training_time_seconds: float