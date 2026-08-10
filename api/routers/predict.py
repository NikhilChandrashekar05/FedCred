# routers/predict.py — loan prediction endpoint
# takes a loan application, runs it through the trained global model
# returns approve/deny decision with SHAP reason codes

from fastapi import APIRouter, HTTPException
import torch
import numpy as np
import pickle
import os
import sys
#import shap

# add the flower_serv_clients folder to path so we can import the model
sys.path.append(os.path.join(os.path.dirname(__file__), '../../flower_serv_clients'))
from model import CreditScoringModel
from schemas import LoanApplication, LoanDecision

router = APIRouter()

# paths to the saved model weights and scaler
# using bank_a's scaler as the default for inference
# in production  use the aggregated global scaler
MODEL_PATH  = os.path.join(os.path.dirname(__file__), '../../flower_serv_clients/global_model.pth')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '../../Data/processed/bank_a_scaler.pkl')

# feature names in the exact order the model was trained on
# used for SHAP reason codes so explanations are readable
FEATURE_NAMES = [
    'loan_amnt', 'int_rate', 'installment', 'annual_inc',
    'dti', 'fico_range_low', 'fico_range_high', 'inq_last_6mths',
    'open_acc', 'pub_rec', 'revol_bal', 'revol_util',
    'total_acc', 'delinq_2yrs', 'emp_length', 'home_ownership'
]

#  readable names for each feature
# shown in the loan decision response instead of raw column names
FEATURE_LABELS = {
    'loan_amnt':       'Loan amount requested',
    'int_rate':        'Interest rate',
    'installment':     'Monthly installment',
    'annual_inc':      'Annual income',
    'dti':             'Debt-to-income ratio',
    'fico_range_low':  'Credit score (low)',
    'fico_range_high': 'Credit score (high)',
    'inq_last_6mths':  'Credit inquiries last 6 months',
    'open_acc':        'Number of open accounts',
    'pub_rec':         'Public derogatory records',
    'revol_bal':       'Revolving credit balance',
    'revol_util':      'Revolving utilization rate',
    'total_acc':       'Total credit accounts',
    'delinq_2yrs':     'Delinquencies in last 2 years',
    'emp_length':      'Employment length',
    'home_ownership':  'Home ownership status',
}

def load_model():
    # load the trained global model weights from disk
    model = CreditScoringModel(input_dim=16)

    if not os.path.exists(MODEL_PATH):
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. Run federated training first."
        )

    # load the saved weights into the model
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))

    # set to eval mode for inference
    model.eval()
    return model

def load_scaler():
    # load the saved MinMaxScaler from disk
    if not os.path.exists(SCALER_PATH):
        raise HTTPException(
            status_code=503,
            detail="Scaler not found. Run preprocessing first."
        )

    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    return scaler

def get_shap_reasons(application, probability):
    # rule-based reason codes ,real SHAP pending tools install
    reasons = []

    if application.dti > 20:
        reasons.append("High debt-to-income ratio negatively impacted this decision")
    if application.fico_range_low < 650:
        reasons.append("Low credit score negatively impacted this decision")
    if application.inq_last_6mths > 2:
        reasons.append("Multiple recent credit inquiries negatively impacted this decision")
    if application.delinq_2yrs > 0:
        reasons.append("Recent delinquencies negatively impacted this decision")
    if application.annual_inc > 80000:
        reasons.append("Strong annual income positively impacted this decision")
    if application.revol_util > 70:
        reasons.append("High revolving credit utilization negatively impacted this decision")

    if len(reasons) == 0:
        reasons = ["Credit profile evaluated", "Income and debt reviewed", "Payment history considered"]

    return reasons[:3]

@router.post("/", response_model=LoanDecision)
def predict_loan(application: LoanApplication):
    # load model and scaler
    model = load_model()
    scaler = load_scaler()

   # separate numeric features (15) from categorical (home_ownership)
    # scaler was fitted on 15 numeric features only during preprocessing
    numeric_input = np.array([[
        application.loan_amnt,
        application.int_rate,
        application.installment,
        application.annual_inc,
        application.dti,
        application.fico_range_low,
        application.fico_range_high,
        application.inq_last_6mths,
        application.open_acc,
        application.pub_rec,
        application.revol_bal,
        application.revol_util,
        application.total_acc,
        application.delinq_2yrs,
        application.emp_length,
    ]], dtype=np.float32)

    # scale the 15 numeric features
    numeric_scaled = scaler.transform(numeric_input)

    # append home_ownership unscaled as the 16th feature
    home_ownership = np.array([[application.home_ownership]], dtype=np.float32)
    input_scaled = np.hstack([numeric_scaled, home_ownership])

    # run through model
    tensor = torch.tensor(input_scaled, dtype=torch.float32)
    with torch.no_grad():
        raw_output = model(tensor)
        probability = torch.sigmoid(raw_output).item()

    # make decision based on 0.5 threshold
    decision = "APPROVED" if probability >= 0.5 else "DENIED"

    # get SHAP reason codes
    reasons = get_shap_reasons(application, probability)

    return LoanDecision(
        decision=decision,
        probability=round(probability, 4),
        reasons=reasons
    )