# 🏏 IPL Intelligence Dashboard

**IPL Intelligence Dashboard** is a season-aware analytics platform for the Indian Premier League (2008–2026). It features dynamic player-team mapping, premium career timelines, a multi-layer team system (active vs. analysis), and match predictors. Built with Streamlit, it offers deep-dive insights into stats and squads.

## 📁 Project Structure
```
ipl_project/
├── app.py                  ← Main Streamlit dashboard
├── requirements.txt        ← Python dependencies
├── ipl_database.db         ← SQLite database (all tables)
├── matches_clean.csv       ← Cleaned match data
├── deliveries_clean.csv    ← Cleaned ball-by-ball data
├── batting_stats.csv       ← Aggregated batting stats
├── bowling_stats.csv       ← Aggregated bowling stats
├── team_stats.csv          ← Team performance + form
├── venue_stats.csv         ← Venue analytics
└── models/
    ├── gb_model.pkl        ← Gradient Boosting (best: ~58%)
    ├── rf_model.pkl        ← Random Forest
    ├── lr_model.pkl        ← Logistic Regression
    ├── h2h_data.pkl        ← Head-to-head history
    ├── team_win_pct.pkl    ← Team win percentages
    ├── team_form.pkl       ← Recent form index
    ├── venue_avg.pkl       ← Venue average scores
    └── season_form.pkl     ← Season-wise form

## 🚀 How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📊 Dashboard Modules
1. **Overview** — KPIs, match trends, toss impact, title winners
2. **Team Analytics** — Win %, H2H, season trends
3. **Player Analytics** — Batting/bowling stats, player comparison
4. **Venue Insights** — Scoring patterns, chase vs defend
5. **Match Predictor** — AI win probability (Gradient Boosting)
6. **Playing XI** — Recommender based on impact scores
7. **IPL 2026 Live** — Manual entry tracker + points table

## 📈 Model Performance (Improved)
| Model | Accuracy (Validated) | Type |
|---|---|---|
| Gradient Boosting (v1) | 58.3%* | Baseline |
| XGBoost (v2) | **54.27%** | **Chronological (Honest)** |

> [!NOTE]
> *v1 accuracy was reported on the full dataset. v2 uses a strict chronological split (Training: 2008-2023, Testing: 2024) to ensure real-world predictive power without data leakage.

## 🗄️ Database Tables (SQLite)
- `matches` — 1,095 match records (2008–2024)
- `deliveries` — 260,920 ball-by-ball records
- `batting_stats` — Aggregated per batsman
- `bowling_stats` — Aggregated per bowler
- `team_stats` — Win %, form index
- `venue_stats` — Avg scores, chase/defend
- `ipl_2026_live` — Manual 2026 entry table
