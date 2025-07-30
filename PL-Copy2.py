#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[56]:


df = pd.read_html('https://fbref.com/en/comps/9/Premier-League-Stats', attrs={"id":"results2024-202591_home_away"})[0]


# In[57]:


df.columns = ['_'.join(col).strip() for col in df.columns.values]


# In[58]:


df.head()


# In[59]:


import pandas as pd
import numpy as np
from scipy.stats import poisson



# In[ ]:


# Rename for clarity (assuming second column is team names)
df.rename(columns={df.columns[1]: "Team"}, inplace=True)
df.set_index("Team", inplace=True)


# In[10]:


# --- Calculate average goals ---
league_avg_home_goals = df["Home_GF"].sum() / df["Home_MP"].sum()
league_avg_away_goals = df["Away_GF"].sum() / df["Away_MP"].sum()


# In[65]:


# --- Define Poisson-based prediction function ---
def calculate_team_strengths(team):
    home_attack = df.loc[team, "Home_GF"] / df.loc[team, "Home_MP"] / league_avg_home_goals
    home_defense = df.loc[team, "Home_GA"] / df.loc[team, "Home_MP"] / league_avg_away_goals
    away_attack = df.loc[team, "Away_GF"] / df.loc[team, "Away_MP"] / league_avg_away_goals
    away_defense = df.loc[team, "Away_GA"] / df.loc[team, "Away_MP"] / league_avg_home_goals
    return home_attack, home_defense, away_attack, away_defense


# In[66]:


def poisson_prediction(home_team, away_team, home_advantage=0.05, injury_handicap_home=0.05, injury_handicap_away=0.10):
    h_attack, h_def, h_away_attack, h_away_def = calculate_team_strengths(home_team)
    a_attack, a_def, a_away_attack, a_away_def = calculate_team_strengths(away_team)

    # Adjusting expected goals
    home_goals_avg = league_avg_home_goals * h_attack * a_def * (1 + home_advantage - injury_handicap_home)
    away_goals_avg = league_avg_away_goals * a_away_attack * h_def * (1 - injury_handicap_away)

    max_goals = 6
    probs = np.zeros((max_goals, max_goals))

    for i in range(max_goals):
        for j in range(max_goals):
            probs[i, j] = poisson.pmf(i, home_goals_avg) * poisson.pmf(j, away_goals_avg)
            # Outcome probabilities
    home_win_prob = np.sum(np.tril(probs, -1))
    draw_prob = np.sum(np.diag(probs))
    away_win_prob = np.sum(np.triu(probs, 1))
    btts_prob = np.sum([probs[i, j] for i in range(1, max_goals) for j in range(1, max_goals)])

    expected_home_goals = sum([i * poisson.pmf(i, home_goals_avg) for i in range(max_goals)])
    expected_away_goals = sum([i * poisson.pmf(i, away_goals_avg) for i in range(max_goals)])

    return {
        "home_win": round(home_win_prob, 3),
        "draw": round(draw_prob, 3),
        "away_win": round(away_win_prob, 3),
        "btts": round(btts_prob, 3),
        "expected_home_goals": round(expected_home_goals, 2),
        "expected_away_goals": round(expected_away_goals, 2),
        "prob_matrix": probs
    }


# In[85]:


if __name__ == "__main__":
    teams = sorted(df.index.tolist())
    print("Available teams:")
    print(", ".join(teams))

    home_team = input("Enter Home Team: ")
    away_team = input("Enter Away Team: ")
    injury_home = float(input("Home Handicap (0.0 to 0.3): "))
    injury_away = float(input("Away Handicap (0.0 to 0.3): "))

    if home_team not in teams or away_team not in teams:
        print("Invalid team name. Please check and try again.")
    elif home_team == away_team:
        print("Teams must be different.")
    else:
        result = poisson_prediction(home_team, away_team, injury_handicap_home=injury_home, injury_handicap_away=injury_away)
        print(f"\nPrediction for {home_team} vs {away_team}:")
        print(f"Home Win Probability: {result['home_win']:.2%}")
        print(f"Draw Probability: {result['draw']:.2%}")
        print(f"Away Win Probability: {result['away_win']:.2%}")
        print(f"Both Teams To Score Probability: {result['btts']:.2%}")
        print(f"Expected Goals - {home_team}: {result['expected_home_goals']}")
        print(f"Expected Goals - {away_team}: {result['expected_away_goals']}")





