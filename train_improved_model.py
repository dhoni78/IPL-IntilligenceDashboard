import pandas as pd
import numpy as np
import os
import pickle
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

matches = pd.read_csv(os.path.join(BASE_DIR, "matches_clean.csv"))
matches['date'] = pd.to_datetime(matches['date'])
matches = matches.sort_values('date').reset_index(drop=True)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
all_teams = list(set(matches['team1'].unique().tolist() + matches['team2'].unique().tolist()))
le.fit(all_teams)

def build_features(df):
    features = []
    team_history, h2h_history, venue_history, vteam_history, team_rolling = {}, {}, {}, {}, {}

    for i, row in df.iterrows():
        t1, t2 = row['team1'], row['team2']
        sorted_teams = tuple(sorted([t1, t2]))
        venue = row['venue']
        
        t1_s = team_history.get(t1, [0, 0]); t2_s = team_history.get(t2, [0, 0])
        h2h_s = h2h_history.get(sorted_teams, [0, 0])
        vt1_s = vteam_history.get((venue, t1), [0, 0]); vt2_s = vteam_history.get((venue, t2), [0, 0])
        v_s = venue_history.get(venue, [0, 0])
        r1 = team_rolling.get(t1, []); r2 = team_rolling.get(t2, [])

        features.append({
            "team1_enc": le.transform([t1])[0],
            "team2_enc": le.transform([t2])[0],
            "t1_wp": (t1_s[0]/t1_s[1]*100) if t1_s[1]>0 else 50.0,
            "t2_wp": (t2_s[0]/t2_s[1]*100) if t2_s[1]>0 else 50.0,
            "t1_form": (sum(r1)/len(r1)*100) if len(r1)>0 else 50.0,
            "t2_form": (sum(r2)/len(r2)*100) if len(r2)>0 else 50.0,
            "h2h_wp": (h2h_s[0]/h2h_s[1]*100 if t1==sorted_teams[0] else (1-h2h_s[0]/h2h_s[1])*100) if h2h_s[1]>0 else 50.0,
            "vt1_wp": (vt1_s[0]/vt1_s[1]*100) if vt1_s[1]>0 else 50.0,
            "vt2_wp": (vt2_s[0]/vt2_s[1]*100) if vt2_s[1]>0 else 50.0,
            "v_avg": (v_s[0]/v_s[1]) if v_s[1]>0 else 165.0,
            "toss_t1": int(row['toss_winner'] == t1),
            "bat_first": int((row['toss_winner'] == t1 and row['toss_decision'] == "bat") or (row['toss_winner'] == t2 and row['toss_decision'] == "field")),
            "winner_is_team1": int(row['winner'] == t1)
        })
        
        if row['winner'] != "No Result":
            tw = row['winner']
            for t in [t1, t2]:
                s = team_history.get(t, [0, 0]); s[1]+=1; s[0]+=(1 if tw==t else 0); team_history[t]=s
                vs = vteam_history.get((venue, t), [0, 0]); vs[1]+=1; vs[0]+=(1 if tw==t else 0); vteam_history[(venue, t)]=vs
                r = team_rolling.get(t, []); r.append(1 if tw==t else 0); team_rolling[t]=r[-5:]
            h2h = h2h_history.get(sorted_teams, [0, 0]); h2h[1]+=1; h2h[0]+=(1 if tw==sorted_teams[0] else 0); h2h_history[sorted_teams]=h2h
        vh = venue_history.get(venue, [0, 0]); vh[1]+=1; vh[0] += row['target_runs'] if row['target_runs']>0 else 165; venue_history[venue] = vh

    return pd.DataFrame(features)

print("🚀 Feature Engineering...")
data = build_features(matches[matches['winner'] != "No Result"])
X, y = data.drop("winner_is_team1", axis=1), data["winner_is_team1"]
split_idx = int(len(data) * 0.85)
X_train, X_test, y_train, y_test = X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

print("🏗️ Training XGBoost (Final Version)...")
model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.03, random_state=42)
model.fit(X_train, y_train)

print(f"✅ Final Test Accuracy: {accuracy_score(y_test, model.predict(X_test))*100:.2f}%")

def calculate_final_metadata(df):
    team_history, h2h_history, venue_history, vteam_history, team_rolling = {}, {}, {}, {}, {}
    for _, row in df.iterrows():
        t1, t2 = row['team1'], row['team2']; sorted_teams = tuple(sorted([t1, t2])); venue = row['venue']
        if row['winner'] != "No Result":
            tw = row['winner']
            for t in [t1, t2]:
                s = team_history.get(t, [0, 0]); s[1]+=1; s[0]+=(1 if tw==t else 0); team_history[t]=s
                vs = vteam_history.get((venue, t), [0, 0]); vs[1]+=1; vs[0]+=(1 if tw==t else 0); vteam_history[(venue, t)]=vs
                r = team_rolling.get(t, []); r.append(1 if tw==t else 0); team_rolling[t]=r[-5:]
            h2h = h2h_history.get(sorted_teams, [0, 0]); h2h[1]+=1; h2h[0]+=(1 if tw==sorted_teams[0] else 0); h2h_history[sorted_teams]=h2h
        vh = venue_history.get(venue, [0, 0]); vh[1]+=1; vh[0] += row['target_runs'] if row['target_runs']>0 else 165; venue_history[venue] = vh
    return ({t: (v[0]/v[1]*100) for t, v in team_history.items() if v[1]>0},
            {t: (sum(v)/len(v)*100) if len(v)>0 else 50.0 for t, v in team_rolling.items()},
            {k: {"total": v[1], "wins": {k[0]: v[0], k[1]: v[1]-v[0]}} for k, v in h2h_history.items()},
            {v: (s[0]/s[1]) for v, s in venue_history.items() if s[1]>0},
            {f"{v}_{t}": (s[0]/s[1]*100) for (v, t), s in vteam_history.items() if s[1]>0})

print("💾 Saving artifacts...")
twp, tf, h2h, va, vt = calculate_final_metadata(matches)
with open(f"{MODELS_DIR}/gb_model_v2.pkl", "wb") as f: pickle.dump(model, f)
with open(f"{MODELS_DIR}/team_win_pct_v2.pkl", "wb") as f: pickle.dump(twp, f)
with open(f"{MODELS_DIR}/team_form_v2.pkl", "wb") as f: pickle.dump(tf, f)
with open(f"{MODELS_DIR}/h2h_data_v2.pkl", "wb") as f: pickle.dump(h2h, f)
with open(f"{MODELS_DIR}/venue_avg_v2.pkl", "wb") as f: pickle.dump(va, f)
with open(f"{MODELS_DIR}/venue_team_stats_v2.pkl", "wb") as f: pickle.dump(vt, f)
with open(f"{MODELS_DIR}/team_encoder.pkl", "wb") as f: pickle.dump(le, f)
print("🏁 Training complete.")
