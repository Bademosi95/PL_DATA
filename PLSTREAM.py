import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
from PL_Copy2 import df, poisson_prediction  # Ensure your model logic is in model.py

# --- Streamlit UI ---
st.set_page_config(page_title="Football Match Predictor", layout="centered")
st.title("⚽ Football Match Outcome Predictor")

st.markdown("""
Select two teams and adjust real-world factors like home advantage or injuries to simulate match outcomes.
""")

teams = df.index.tolist()

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Home Team", teams)
with col2:
    away_team = st.selectbox("Away Team", [team for team in teams if team != home_team])

st.subheader("Adjust Match Conditions")
home_advantage = st.slider("Home Advantage (Strong Home Form)", 0.0, 0.5, 0.1, step=0.05)
injury_handicap_home = st.slider("Home Team Handicap (e.g. Injuries, Form)", 0.0, 1.0, 0.0, step=0.1)
injury_handicap_away = st.slider("Away Team Handicap (e.g. Injuries, Form)", 0.0, 1.0, 0.0, step=0.1)

if st.button("Predict Outcome"):
    result = poisson_prediction(
        home_team=home_team,
        away_team=away_team,
        home_advantage=home_advantage,
        injury_handicap_home=injury_handicap_home,
        injury_handicap_away=injury_handicap_away
    )

    st.success("Match Simulation Complete")

    st.subheader("Predicted Score and Probabilities")
    st.markdown(f"**Expected Goals**: {home_team}: {result['expected_home_goals']} | {away_team}: {result['expected_away_goals']}")
    st.markdown(f"- **{home_team} Win**: {result['home_win']*100:.1f}%")
    st.markdown(f"- **Draw**: {result['draw']*100:.1f}%")
    st.markdown(f"- **{away_team} Win**: {result['away_win']*100:.1f}%")
    st.markdown(f"- **Both Teams to Score**: {result['btts']*100:.1f}%")