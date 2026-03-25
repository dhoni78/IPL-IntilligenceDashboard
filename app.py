
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import date
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IPL Intelligence Dashboard",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── TEAM CLASSIFICATION ───────────────────────────────────────────────────────
CURRENT_VIEW = [
    "Chennai Super Kings", "Mumbai Indians", "Royal Challengers Bengaluru",
    "Kolkata Knight Riders", "Delhi Capitals", "Punjab Kings",
    "Rajasthan Royals", "Sunrisers Hyderabad", "Lucknow Super Giants", "Gujarat Titans"
]

DEFUNCT_TEAMS = [
    "Deccan Chargers", "Gujarat Lions", "Kochi Tuskers Kerala",
    "Pune Warriors India", "Rising Pune Supergiant"
]

ANALYSIS_POOL = sorted(list(set(CURRENT_VIEW + DEFUNCT_TEAMS)))

TEAM_NORMALIZATION = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Pune Warriors": "Pune Warriors India",
    "DC": "Delhi Capitals", "PBKS": "Punjab Kings", "RCB": "Royal Challengers Bengaluru",
    "CSK": "Chennai Super Kings", "MI": "Mumbai Indians", "KKR": "Kolkata Knight Riders",
    "RR": "Rajasthan Royals", "SRH": "Sunrisers Hyderabad", "GT": "Gujarat Titans", "LSG": "Lucknow Super Giants"
}

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f, #0d2137);
        border-radius: 12px; padding: 20px; text-align: center;
        border: 1px solid #1a4a7a; margin: 5px;
    }
    .metric-val { font-size: 2rem; font-weight: bold; color: #00d4ff; }
    .metric-lbl { font-size: 0.85rem; color: #aaa; }
    .win-bar-team1 { background: #00d4ff; height: 28px; border-radius: 4px 0 0 4px; }
    .win-bar-team2 { background: #ff6b35; height: 28px; border-radius: 0 4px 4px 0; }
    [data-testid="stSidebar"] { background-color: #111827; }
    h1, h2, h3 { color: #e0e0e0; }
    .stTabs [data-baseweb="tab"] { color: #aaa; font-size: 15px; }
    .stTabs [aria-selected="true"] { color: #00d4ff; border-bottom: 2px solid #00d4ff; }
</style>
""", unsafe_allow_html=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_data():
    conn = sqlite3.connect(os.path.join(BASE_DIR, "ipl_database.db"))
    matches    = pd.read_sql("SELECT * FROM matches",       conn)
    bat        = pd.read_sql("SELECT * FROM batting_stats", conn)
    bowl       = pd.read_sql("SELECT * FROM bowling_stats", conn)
    team_stats = pd.read_sql("SELECT * FROM team_stats",    conn)
    venue_stats= pd.read_sql("SELECT * FROM venue_stats",   conn)
    live_2026  = pd.read_sql("SELECT * FROM ipl_2026_live", conn)
    mapping    = pd.read_sql("SELECT * FROM player_team_mapping", conn)
    conn.close()

    # Apply Normalization
    for df in [matches, team_stats, mapping, live_2026]:
        for col in ["team1", "team2", "winner", "team", "batting_team", "bowling_team"]:
            if col in df.columns:
                df[col] = df[col].replace(TEAM_NORMALIZATION)

    # Filter for Analysis Pool (keep all teams needed for statistics/ML)
    matches = matches[matches["team1"].isin(ANALYSIS_POOL) & matches["team2"].isin(ANALYSIS_POOL)]
    team_stats = team_stats[team_stats["team"].isin(ANALYSIS_POOL)]
    mapping = mapping[mapping["team"].isin(ANALYSIS_POOL)]
    live_2026 = live_2026[live_2026["team1"].isin(ANALYSIS_POOL) | live_2026["team2"].isin(ANALYSIS_POOL)]

    matches["date"] = pd.to_datetime(matches["date"])
    deliveries = pd.read_csv(os.path.join(BASE_DIR, "deliveries_clean.csv"))
    
    # Normalize and filter deliveries for analysis
    deliveries["batting_team"] = deliveries["batting_team"].replace(TEAM_NORMALIZATION)
    deliveries["bowling_team"] = deliveries["bowling_team"].replace(TEAM_NORMALIZATION)
    deliveries = deliveries[deliveries["batting_team"].isin(ANALYSIS_POOL) & deliveries["bowling_team"].isin(ANALYSIS_POOL)]

    return matches, bat, bowl, team_stats, venue_stats, live_2026, deliveries, mapping

@st.cache_resource
def load_models():
    base = os.path.join(BASE_DIR, "models")
    with open(f"{base}/gb_model_v2.pkl","rb")       as f: gb  = pickle.load(f)
    with open(f"{base}/team_win_pct_v2.pkl","rb")   as f: twp = pickle.load(f)
    with open(f"{base}/team_form_v2.pkl","rb")      as f: tf  = pickle.load(f)
    with open(f"{base}/venue_avg_v2.pkl","rb")      as f: va  = pickle.load(f)
    with open(f"{base}/h2h_data_v2.pkl","rb")       as f: h2h = pickle.load(f)
    with open(f"{base}/venue_team_stats_v2.pkl","rb") as f: vts = pickle.load(f)
    with open(f"{base}/team_encoder.pkl","rb")      as f: le  = pickle.load(f)
    return gb, twp, tf, va, h2h, vts, le

matches, bat, bowl, team_stats, venue_stats, live_2026, deliveries, mapping = load_data()
gb_model, team_win_pct, team_form, venue_avg, h2h_data, venue_team_stats, team_encoder = load_models()

ALL_TEAMS = CURRENT_VIEW
ALL_VENUES = sorted(matches["venue"].unique())
SEASONS = sorted(matches["season"].unique())
MAPPING_SEASONS = sorted(mapping["season"].unique())

# Get all unique player names for autocomplete
ALL_PLAYERS = sorted(list(set(
    deliveries["batter"].dropna().unique().tolist() + 
    deliveries["bowler"].dropna().unique().tolist() + 
    matches["player_of_match"].dropna().unique().tolist()
)))

# ── HELPER FUNCTIONS ─────────────────────────────────────────────────────────
def get_h2h_pct(t1, t2):
    key = tuple(sorted([t1, t2]))
    if key not in h2h_data or h2h_data[key]["total"] == 0: return 50.0
    wins = h2h_data[key]["wins"].get(t1, 0)
    return round(wins / h2h_data[key]["total"] * 100, 2)

def build_pred_features(t1, t2, venue, toss_winner, toss_decision):
    vt1_key = f"{venue}_{t1}"
    vt2_key = f"{venue}_{t2}"
    return pd.DataFrame([{
        "team1_enc":       team_encoder.transform([t1])[0],
        "team2_enc":       team_encoder.transform([t2])[0],
        "t1_wp":           team_win_pct.get(t1, 50),
        "t2_wp":           team_win_pct.get(t2, 50),
        "t1_form":         team_form.get(t1, 50),
        "t2_form":         team_form.get(t2, 50),
        "h2h_wp":          get_h2h_pct(t1, t2),
        "vt1_wp":          venue_team_stats.get(vt1_key, 50),
        "vt2_wp":          venue_team_stats.get(vt2_key, 50),
        "v_avg":           venue_avg.get(venue, 165),
        "toss_t1":         int(toss_winner == t1),
        "bat_first":       int((toss_winner == t1 and toss_decision == "bat") or (toss_winner == t2 and toss_decision == "field"))
    }])

def color_team(team):
    colors = {
        "Mumbai Indians": "#004BA0", "Chennai Super Kings": "#FFFF00",
        "Royal Challengers Bengaluru": "#D11B22", "Kolkata Knight Riders": "#3A225D",
        "Punjab Kings": "#DDAABB", "Rajasthan Royals": "#254AA5",
        "Sunrisers Hyderabad": "#F7A721", "Delhi Capitals": "#0080E8",
        "Gujarat Titans": "#1C1C5E", "Lucknow Super Giants": "#A0E6FF",
    }
    return colors.get(team, "#555555")

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://www.iplt20.com/assets/images/ipl-logo-new-old.png", width=120)
st.sidebar.title("🏏 IPL Intelligence")
page = st.sidebar.radio("Navigate", [
    "📊 Overview", "🏆 Team Analytics", "👥 Player Roster", "👤 Player Analytics",
    "🏟️ Venue Insights", "🤖 Match Predictor", "🎯 Playing XI", "🟢 IPL 2026 Live"
])
st.sidebar.markdown("---")
season_filter = st.sidebar.multiselect("Filter Season", SEASONS, default=SEASONS)
filtered = matches[matches["season"].isin(season_filter)]

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 IPL Match Intelligence Dashboard")
    st.caption(f"Covering {int(filtered['season'].min())}–{int(filtered['season'].max())} | {len(filtered):,} matches | {len(deliveries):,} deliveries")
    st.markdown("---")

    # KPI metrics
    c1,c2,c3,c4,c5 = st.columns(5)
    valid = filtered[filtered["winner"] != "No Result"]
    c1.metric("Total Matches",    f"{len(filtered):,}")
    c2.metric("Total Runs",       f"{deliveries['total_runs'].sum():,}")
    c3.metric("Total Wickets",    f"{deliveries['is_wicket'].sum():,}")
    c4.metric("Highest Target",   f"{filtered['target_runs'].max()}")
    c5.metric("Avg Match Score",  f"{int(filtered['target_runs'][filtered['target_runs']>0].mean())}")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏅 Team Win Count (All Time)")
        wins = valid["winner"].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(7,5), facecolor="#111")
        bars = ax.barh(wins.index[::-1], wins.values[::-1],
                       color=[color_team(t) for t in wins.index[::-1]])
        ax.set_facecolor("#111"); ax.tick_params(colors="white")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        for spine in ["left","bottom"]: ax.spines[spine].set_color("#333")
        ax.set_xlabel("Wins", color="white"); ax.xaxis.label.set_color("white")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("📅 Matches Per Season")
        mps = filtered.groupby("season").size().reset_index(name="matches")
        fig, ax = plt.subplots(figsize=(7,5), facecolor="#111")
        ax.plot(mps["season"], mps["matches"], color="#00d4ff", lw=2, marker="o", ms=5)
        ax.fill_between(mps["season"], mps["matches"], alpha=0.2, color="#00d4ff")
        ax.set_facecolor("#111"); ax.tick_params(colors="white")
        for spine in ["top","right"]: ax.spines[spine].set_visible(False)
        for spine in ["left","bottom"]: ax.spines[spine].set_color("#333")
        ax.set_xlabel("Season", color="white"); ax.set_ylabel("Matches", color="white")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("🎲 Toss Decision Impact")
        toss = valid.groupby("toss_decision").agg(
            total=("id","count"), won=("toss_match_winner","sum")).reset_index()
        toss["lost"] = toss["total"] - toss["won"]
        fig, ax = plt.subplots(figsize=(5,4), facecolor="#111")
        x = np.arange(len(toss)); w = 0.35
        ax.bar(x - w/2, toss["won"],  w, label="Also Won Match", color="#00d4ff")
        ax.bar(x + w/2, toss["lost"], w, label="Lost Match",      color="#ff6b35")
        ax.set_xticks(x); ax.set_xticklabels(toss["toss_decision"], color="white")
        ax.set_facecolor("#111"); ax.tick_params(colors="white")
        ax.legend(facecolor="#222", labelcolor="white")
        for spine in ["top","right"]: ax.spines[spine].set_visible(False)
        for spine in ["left","bottom"]: ax.spines[spine].set_color("#333")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col4:
        st.subheader("🏆 Title Winners")
        finals = valid[valid["match_type"].str.lower() == "final"]
        if len(finals):
            champs = finals["winner"].value_counts().reset_index()
            champs.columns = ["Team","Titles"]
            # Only show active franchises in the titles table
            champs = champs[champs["Team"].isin(CURRENT_VIEW)]
            st.dataframe(champs, use_container_width=True, hide_index=True)
        else:
            st.info("No Final matches found in filtered data.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: TEAM ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏆 Team Analytics":
    st.title("🏆 Team Analytics")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Win % by Team")
        ts = team_stats.sort_values("win_pct", ascending=True).tail(12)
        fig, ax = plt.subplots(figsize=(7,6), facecolor="#111")
        colors = [color_team(t) for t in ts["team"]]
        ax.barh(ts["team"], ts["win_pct"], color=colors)
        ax.set_facecolor("#111"); ax.tick_params(colors="white")
        ax.set_xlabel("Win %", color="white")
        for spine in ["top","right"]: ax.spines[spine].set_visible(False)
        for spine in ["left","bottom"]: ax.spines[spine].set_color("#333")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("🔥 Recent Form (Last 5 Matches %)")
        rf = team_stats[team_stats["matches_played"] >= 20].sort_values("recent_form_pct", ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(7,6), facecolor="#111")
        bar_colors = ["#00d4ff" if v >= 60 else "#ff6b35" if v < 40 else "#ffd700" for v in rf["recent_form_pct"]]
        ax.barh(rf["team"][::-1], rf["recent_form_pct"][::-1], color=bar_colors[::-1])
        ax.axvline(50, color="white", linestyle="--", alpha=0.4)
        ax.set_facecolor("#111"); ax.tick_params(colors="white")
        ax.set_xlabel("Win % in last 5 matches", color="white")
        for spine in ["top","right"]: ax.spines[spine].set_visible(False)
        for spine in ["left","bottom"]: ax.spines[spine].set_color("#333")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.subheader("⚔️ Head-to-Head Matchup")
    c1, c2 = st.columns(2)
    t1 = c1.selectbox("Team 1", ALL_TEAMS, index=ALL_TEAMS.index("Mumbai Indians") if "Mumbai Indians" in ALL_TEAMS else 0)
    t2 = c2.selectbox("Team 2", [t for t in ALL_TEAMS if t != t1],
                      index=[t for t in ALL_TEAMS if t != t1].index("Chennai Super Kings")
                      if "Chennai Super Kings" in [t for t in ALL_TEAMS if t != t1] else 0)

    h2h_matches = filtered[
        ((filtered["team1"]==t1)&(filtered["team2"]==t2)) |
        ((filtered["team1"]==t2)&(filtered["team2"]==t1))
    ]
    if len(h2h_matches):
        t1w = (h2h_matches["winner"]==t1).sum()
        t2w = (h2h_matches["winner"]==t2).sum()
        nr  = (h2h_matches["winner"]=="No Result").sum()
        mc1,mc2,mc3,mc4 = st.columns(4)
        mc1.metric("Total Matches", len(h2h_matches))
        mc2.metric(f"{t1} Wins", t1w)
        mc3.metric(f"{t2} Wins", t2w)
        mc4.metric("No Result", nr)
        total = t1w + t2w
        if total > 0:
            p1 = t1w / total * 100
            st.markdown(f"**Win Share:**")
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:8px'>
            <span style='color:#00d4ff;width:120px'>{t1}</span>
            <div style='flex:1;background:#222;border-radius:8px;overflow:hidden;height:28px;display:flex'>
              <div style='width:{p1:.0f}%;background:#00d4ff'></div>
              <div style='width:{100-p1:.0f}%;background:#ff6b35'></div>
            </div>
            <span style='color:#ff6b35;width:120px;text-align:right'>{t2}</span>
            </div>
            <div style='display:flex;justify-content:space-between;color:#aaa;font-size:12px;margin-top:4px'>
              <span>{p1:.1f}%</span><span>{100-p1:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No head-to-head matches found in selected seasons.")

    st.subheader("📊 Season-wise Win % Trend")
    sel_teams = st.multiselect("Select Teams", ALL_TEAMS, default=["Mumbai Indians","Chennai Super Kings"])
    if sel_teams:
        fig, ax = plt.subplots(figsize=(12,5), facecolor="#111")
        for team in sel_teams:
            data = []
            for s in SEASONS:
                sm = matches[matches["season"]==s]
                played = len(sm[(sm["team1"]==team)|(sm["team2"]==team)])
                won    = len(sm[sm["winner"]==team])
                if played: data.append({"season":s,"win_pct":won/played*100})
            if data:
                d = pd.DataFrame(data)
                ax.plot(d["season"], d["win_pct"], marker="o", label=team, lw=2)
        ax.axhline(50, color="white", linestyle="--", alpha=0.3)
        ax.set_facecolor("#111"); ax.tick_params(colors="white")
        ax.legend(facecolor="#222", labelcolor="white")
        ax.set_ylabel("Win %", color="white"); ax.set_xlabel("Season", color="white")
        for spine in ["top","right"]: ax.spines[spine].set_visible(False)
        for spine in ["left","bottom"]: ax.spines[spine].set_color("#333")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: PLAYER ROSTER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥 Player Roster":
    st.title("👥 Player Roster & Career Timeline")
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["📋 Season Roster", "🧬 Career Timeline", "🔄 Transfer Tracker"])
    
    with tab1:
        c1, c2 = st.columns(2)
        r_season = c1.selectbox("Select Season", MAPPING_SEASONS, index=len(MAPPING_SEASONS)-1)
        # Filter teams available in that season
        teams_in_season = sorted(mapping[mapping["season"]==r_season]["team"].unique())
        r_team = c2.selectbox("Select Team", teams_in_season)
        
        roster = mapping[(mapping["season"]==r_season) & (mapping["team"]==r_team)].sort_values("player_name")
        st.subheader(f"Squad: {r_team} ({r_season})")
        
        # Display as a clean list or table
        st.dataframe(roster[["player_name"]].rename(columns={"player_name":"Player Name"}).reset_index(drop=True), 
                     use_container_width=True, hide_index=True)
        
        st.caption(f"Total Squad Size: {len(roster)} players")

    with tab2:
        st.subheader("🧬 Player Career Timeline")
        p_name = st.selectbox("Search Player", sorted(mapping["player_name"].unique()))
        p_history = mapping[mapping["player_name"]==p_name].sort_values("season")
        
        if len(p_history):
            fig, ax = plt.subplots(figsize=(10, min(max(len(p_history)*0.5, 3), 10)), facecolor="#111")
            ax.set_facecolor("#111")
            
            teams = p_history["team"].tolist()
            seasons = p_history["season"].tolist()
            y_pos = np.arange(len(seasons))
            
            # Bar representing time at team
            ax.barh(y_pos, [1]*len(seasons), color=[color_team(t) for t in teams], height=0.6)
            
            # Label with team name
            for i, (s, t) in enumerate(zip(seasons, teams)):
                text_color = "black" if color_team(t).lower() in ["#ffff00", "yellow"] else "white"
                ax.text(0.5, i, f"{t}", ha='center', va='center', color=text_color, fontweight='bold')

            ax.set_yticks(y_pos)
            ax.set_yticklabels(seasons, color="white", fontsize=12)
            ax.set_xticks([])
            ax.invert_yaxis()
            
            for spine in ax.spines.values(): spine.set_visible(False)
            plt.tight_layout()
            st.pyplot(fig); plt.close()
            
            st.table(p_history[["season", "team"]].rename(columns={"season":"Season", "team":"Team"}).reset_index(drop=True))
        else:
            st.warning("No career history found for this player.")

    with tab3:
        st.subheader("🔄 Major Transfers")
        t_s1 = st.selectbox("From Season", MAPPING_SEASONS[:-1], index=len(MAPPING_SEASONS)-2)
        t_s2 = t_s1 + 1
        st.info(f"Showing players who changed teams between **{t_s1}** and **{t_s2}**")
        
        m1 = mapping[mapping["season"]==t_s1][["player_name", "team"]].rename(columns={"team": "Old Team"})
        m2 = mapping[mapping["season"]==t_s2][["player_name", "team"]].rename(columns={"team": "New Team"})
        
        transfers = pd.merge(m1, m2, on="player_name")
        transfers = transfers[transfers["Old Team"] != transfers["New Team"]]
        
        if not transfers.empty:
            st.dataframe(transfers.rename(columns={"player_name":"Player"}).reset_index(drop=True), 
                         use_container_width=True, hide_index=True)
            st.metric("Total Transfers", len(transfers))
        else:
            st.info(f"No transfers recorded between {t_s1} and {t_s2}.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: PLAYER ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👤 Player Analytics":
    st.title("👤 Player Analytics")
    tab1, tab2, tab3 = st.tabs(["🏏 Batting", "🎳 Bowling", "🔍 Player Compare"])

    with tab1:
        c1,c2 = st.columns([1,3])
        min_balls = c1.slider("Min balls faced", 50, 500, 100)
        top_n     = c1.slider("Top N players", 5, 20, 10)
        metric    = c1.selectbox("Sort by", ["total_runs","strike_rate","batting_average","sixes"])
        
        # Season filter for stats
        s_filter = c1.selectbox("Filter by Season (Roster)", ["All Time"] + [str(s) for s in reversed(MAPPING_SEASONS)])
        if s_filter != "All Time":
            season_players = mapping[mapping["season"] == int(s_filter)]["player_name"].unique()
            filtered_bat = bat[bat["batter"].isin(season_players)]
        else:
            filtered_bat = bat
            
        bd = filtered_bat[filtered_bat["balls_faced"] >= min_balls].nlargest(top_n, metric)
        with c2:
            st.subheader(f"Top {top_n} Batsmen — {metric.replace('_',' ').title()}")
            fig, ax = plt.subplots(figsize=(8,5), facecolor="#111")
            ax.barh(bd["batter"][::-1], bd[metric][::-1], color="#00d4ff")
            ax.set_facecolor("#111"); ax.tick_params(colors="white")
            ax.set_xlabel(metric.replace("_"," ").title(), color="white")
            for spine in ["top","right"]: ax.spines[spine].set_visible(False)
            for spine in ["left","bottom"]: ax.spines[spine].set_color("#333")
            plt.tight_layout(); st.pyplot(fig); plt.close()
        st.dataframe(bd[["batter","total_runs","balls_faced","strike_rate","batting_average","fours","sixes","bat_score"]].reset_index(drop=True),
                     use_container_width=True, hide_index=True)

    with tab2:
        c1,c2 = st.columns([1,3])
        min_balls_b = c1.slider("Min balls bowled", 100, 1000, 200)
        top_nb      = c1.slider("Top N bowlers", 5, 20, 10)
        metric_b    = c1.selectbox("Sort by", ["wickets","economy","bowling_avg","dot_balls"])
        
        if s_filter != "All Time":
            season_players = mapping[mapping["season"] == int(s_filter)]["player_name"].unique()
            filtered_bowl = bowl[bowl["bowler"].isin(season_players)]
        else:
            filtered_bowl = bowl
            
        bd2 = filtered_bowl[filtered_bowl["balls_bowled"] >= min_balls_b]
        bd2 = bd2.nsmallest(top_nb, metric_b) if metric_b in ["economy","bowling_avg"] else bd2.nlargest(top_nb, metric_b)
        with c2:
            st.subheader(f"Top {top_nb} Bowlers — {metric_b.replace('_',' ').title()}")
            fig, ax = plt.subplots(figsize=(8,5), facecolor="#111")
            ax.barh(bd2["bowler"][::-1], bd2[metric_b][::-1], color="#ff6b35")
            ax.set_facecolor("#111"); ax.tick_params(colors="white")
            ax.set_xlabel(metric_b.replace("_"," ").title(), color="white")
            for spine in ["top","right"]: ax.spines[spine].set_visible(False)
            for spine in ["left","bottom"]: ax.spines[spine].set_color("#333")
            plt.tight_layout(); st.pyplot(fig); plt.close()
        st.dataframe(bd2[["bowler","wickets","overs","economy","bowling_avg","dot_balls","bowl_score"]].reset_index(drop=True),
                     use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("🔍 Player Comparison Tool")
        pc1, pc2 = st.columns(2)
        
        # Search assist: filter by team/season
        with st.expander("🔍 Search Assist (Filter by Team/Season)"):
            sc1, sc2 = st.columns(2)
            search_s = sc1.selectbox("Season", ["All Time"] + [str(s) for s in reversed(MAPPING_SEASONS)], key="search_s")
            if search_s != "All Time":
                available_teams = sorted(mapping[mapping["season"]==int(search_s)]["team"].unique())
                search_t = sc2.selectbox("Team", ["All Teams"] + available_teams, key="search_t")
                
                search_mapping = mapping[mapping["season"]==int(search_s)]
                if search_t != "All Teams":
                    search_mapping = search_mapping[search_mapping["team"] == search_t]
                
                player_list = sorted(search_mapping["player_name"].unique())
            else:
                player_list = sorted(bat["batter"].tolist())
        
        p1 = pc1.selectbox("Player 1", player_list, key="p1")
        p2 = pc2.selectbox("Player 2", player_list, key="p2", 
                           index=min(1, len(player_list)-1) if len(player_list)>1 else 0)
        p1d = bat[bat["batter"]==p1].iloc[0] if len(bat[bat["batter"]==p1]) else None
        p2d = bat[bat["batter"]==p2].iloc[0] if len(bat[bat["batter"]==p2]) else None
        if p1d is not None and p2d is not None:
            metrics_cmp = ["total_runs","strike_rate","batting_average","fours","sixes","bat_score"]
            labels_cmp  = ["Total Runs","Strike Rate","Avg","4s","6s","Impact Score"]
            fig, axes = plt.subplots(1, len(metrics_cmp), figsize=(14,4), facecolor="#111")
            for i, (m, l) in enumerate(zip(metrics_cmp, labels_cmp)):
                axes[i].bar([p1, p2], [p1d[m], p2d[m]], color=["#00d4ff","#ff6b35"])
                axes[i].set_title(l, color="white", fontsize=10)
                axes[i].set_facecolor("#111"); axes[i].tick_params(colors="white", labelsize=7)
                for spine in ["top","right"]: axes[i].spines[spine].set_visible(False)
                for spine in ["left","bottom"]: axes[i].spines[spine].set_color("#333")
            plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: VENUE INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏟️ Venue Insights":
    st.title("🏟️ Venue Insights")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏟️ Highest Scoring Venues")
        vs = venue_stats.dropna().sort_values("avg_target", ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(8,5), facecolor="#111")
        ax.barh([v[:30] for v in vs["venue"]][::-1], vs["avg_target"][::-1], color="#ffd700")
        ax.set_facecolor("#111"); ax.tick_params(colors="white")
        ax.set_xlabel("Avg First Innings Target", color="white")
        for spine in ["top","right"]: ax.spines[spine].set_visible(False)
        for spine in ["left","bottom"]: ax.spines[spine].set_color("#333")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.subheader("🎯 Chase vs Defend at Venues")
        venue_sel = st.selectbox("Select Venue", sorted(filtered["venue"].unique()))
        vm = filtered[filtered["venue"]==venue_sel]
        bat_first  = vm[vm["toss_decision"]=="bat"]
        field_first= vm[vm["toss_decision"]=="field"]
        chased_won = field_first[field_first["toss_match_winner"]==1]
        defend_won = bat_first[bat_first["toss_match_winner"]==1]
        mc1,mc2,mc3 = st.columns(3)
        mc1.metric("Total Matches", len(vm))
        mc2.metric("Chasing Won",   len(chased_won))
        mc3.metric("Defending Won", len(defend_won))
        if len(chased_won)+len(defend_won) > 0:
            labels = ["Chasing Win", "Defending Win"]
            vals   = [len(chased_won), len(defend_won)]
            fig, ax = plt.subplots(figsize=(4,4), facecolor="#111")
            wedges, texts, at = ax.pie(vals, labels=labels, autopct="%1.0f%%",
                                       colors=["#00d4ff","#ff6b35"],
                                       textprops={"color":"white"},
                                       startangle=90)
            ax.set_facecolor("#111")
            plt.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("📊 Phase-wise Scoring by Venue")
    sel_venue = st.selectbox("Venue for phase analysis", sorted(filtered["venue"].unique()), key="phase_v")
    vm_ids = filtered[filtered["venue"]==sel_venue]["id"].tolist()
    vd = deliveries[deliveries["match_id"].isin(vm_ids)]
    if len(vd):
        phase_runs = vd.groupby("phase")["total_runs"].mean().reindex(["Powerplay","Middle","Death"])
        fig, ax = plt.subplots(figsize=(6,4), facecolor="#111")
        ax.bar(phase_runs.index, phase_runs.values,
               color=["#00d4ff","#ffd700","#ff6b35"])
        ax.set_facecolor("#111"); ax.tick_params(colors="white")
        ax.set_ylabel("Avg Runs/Ball", color="white")
        for spine in ["top","right"]: ax.spines[spine].set_visible(False)
        for spine in ["left","bottom"]: ax.spines[spine].set_color("#333")
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: MATCH PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Match Predictor":
    st.title("🤖 AI Match Predictor")
    st.caption("Powered by Gradient Boosting — trained on 2008–2024 IPL data")
    st.markdown("---")

    c1,c2,c3 = st.columns(3)
    with c1:
        team1 = st.selectbox("🔵 Team 1", ALL_TEAMS)
    with c2:
        team2 = st.selectbox("🔴 Team 2", [t for t in ALL_TEAMS if t != team1])
    with c3:
        venue = st.selectbox("🏟️ Venue", ALL_VENUES)

    c4,c5,c6 = st.columns(3)
    with c4:
        toss_winner = st.selectbox("🎲 Toss Winner", [team1, team2])
    with c5:
        toss_decision = st.selectbox("📋 Toss Decision", ["field","bat"])
    with c6:
        season = st.selectbox("📅 Season", [2024, 2025, 2026])

    if st.button("🔮 Predict Match Outcome", use_container_width=True):
        X_pred = build_pred_features(team1, team2, venue, toss_winner, toss_decision)
        prob   = gb_model.predict_proba(X_pred)[0]
        t1_prob = prob[1] * 100
        t2_prob = prob[0] * 100
        winner  = team1 if t1_prob > t2_prob else team2
        conf    = max(t1_prob, t2_prob)

        st.markdown("---")
        st.subheader("🏆 Prediction Result")
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric(team1, f"{t1_prob:.1f}%", delta=f"{t1_prob-50:.1f}% vs baseline")
        rc2.metric("Predicted Winner", winner)
        rc3.metric(team2, f"{t2_prob:.1f}%", delta=f"{t2_prob-50:.1f}% vs baseline")

        st.markdown(f"""
        <div style='background:#1a2a3a;border-radius:12px;padding:16px;margin-top:12px'>
        <div style='display:flex;align-items:center;gap:8px'>
          <span style='color:#00d4ff;width:160px;font-weight:bold'>{team1}</span>
          <div style='flex:1;background:#222;border-radius:8px;overflow:hidden;height:32px;display:flex'>
            <div style='width:{t1_prob:.1f}%;background:linear-gradient(90deg,#00d4ff,#0080ff)'></div>
            <div style='width:{t2_prob:.1f}%;background:linear-gradient(90deg,#ff6b35,#cc2200)'></div>
          </div>
          <span style='color:#ff6b35;width:160px;text-align:right;font-weight:bold'>{team2}</span>
        </div>
        <div style='display:flex;justify-content:space-between;color:#aaa;font-size:12px;margin-top:6px;padding:0 160px'>
          <span>{t1_prob:.1f}%</span><span>{t2_prob:.1f}%</span>
        </div>
        </div>
        """, unsafe_allow_html=True)

        key = tuple(sorted([team1, team2]))
        if key in h2h_data and h2h_data[key]["total"] > 0:
            h2h_t1 = h2h_data[key]["wins"].get(team1, 0)
            h2h_t2 = h2h_data[key]["wins"].get(team2, 0)
            st.info(f"📊 Historical H2H: **{team1}** won {h2h_t1} | **{team2}** won {h2h_t2} (out of {h2h_data[key]['total']} meetings)")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6: PLAYING XI RECOMMENDER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Playing XI":
    st.title("🎯 Playing XI Recommender")
    st.caption("Suggests best 11 based on overall impact scores")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        sel_team = st.selectbox("Select Team", ALL_TEAMS)
        venue_xi = st.selectbox("Match Venue", ALL_VENUES)

    team_deliveries = deliveries[(deliveries["batting_team"]==sel_team) | (deliveries["bowling_team"]==sel_team)]
    
    # Use mapping for squad roster
    squad_season = st.selectbox("Squad Season", reversed(MAPPING_SEASONS), index=0)
    squad = mapping[(mapping["team"]==sel_team) & (mapping["season"]==squad_season)]["player_name"].tolist()
    
    if squad:
        team_batters = [p for p in squad if p in bat["batter"].values]
        team_bowlers = [p for p in squad if p in bowl["bowler"].values]
        
        # Ensure we have enough players for the recommender
        bat_xi  = bat[bat["batter"].isin(team_batters)].nlargest(7, "bat_score")
        bowl_xi = bowl[bowl["bowler"].isin(team_bowlers)].nlargest(4, "bowl_score")
    else:
        # Fallback to historical data if no mapping found for that team/season
        team_batters = deliveries[deliveries["batting_team"]==sel_team]["batter"].value_counts().head(20).index.tolist()
        team_bowlers = deliveries[deliveries["bowling_team"]==sel_team]["bowler"].value_counts().head(15).index.tolist()
        bat_xi  = bat[bat["batter"].isin(team_batters)].nlargest(7,"bat_score")
        bowl_xi = bowl[bowl["bowler"].isin(team_bowlers)].nlargest(4,"bowl_score")

    st.subheader(f"🏏 Recommended Playing XI for {sel_team}")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Top 7 Batsmen**")
        for i, (_, row) in enumerate(bat_xi.iterrows(), 1):
            st.markdown(f"`{i}.` **{row['batter']}** — Runs: {row['total_runs']:,} | SR: {row['strike_rate']} | Score: {row['bat_score']:.1f}")
    with col4:
        st.markdown("**Top 4 Bowlers**")
        for i, (_, row) in enumerate(bowl_xi.iterrows(), 1):
            st.markdown(f"`{i}.` **{row['bowler']}** — Wkts: {row['wickets']} | Econ: {row['economy']} | Score: {row['bowl_score']:.1f}")

    st.markdown("---")
    st.subheader("📊 Batting vs Bowling Balance")
    bc1, bc2 = st.columns(2)
    with bc1:
        fig, ax = plt.subplots(figsize=(6,4), facecolor="#111")
        ax.barh(bat_xi["batter"][::-1], bat_xi["bat_score"][::-1], color="#00d4ff")
        ax.set_facecolor("#111"); ax.tick_params(colors="white")
        ax.set_title("Batting Impact", color="white")
        for spine in ["top","right"]: ax.spines[spine].set_visible(False)
        for spine in ["left","bottom"]: ax.spines[spine].set_color("#333")
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with bc2:
        fig, ax = plt.subplots(figsize=(6,4), facecolor="#111")
        ax.barh(bowl_xi["bowler"][::-1], bowl_xi["bowl_score"][::-1], color="#ff6b35")
        ax.set_facecolor("#111"); ax.tick_params(colors="white")
        ax.set_title("Bowling Impact", color="white")
        for spine in ["top","right"]: ax.spines[spine].set_visible(False)
        for spine in ["left","bottom"]: ax.spines[spine].set_color("#333")
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7: IPL 2026 LIVE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🟢 IPL 2026 Live":
    st.title("🟢 IPL 2026 Live Tracker")
    st.caption("Manually add match results to track the 2026 season")
    st.markdown("---")

    tab1, tab2 = st.tabs(["➕ Add Match Result", "📊 Points Table & Stats"])

    with tab1:
        st.subheader("Add IPL 2026 Match Result")
        lc1,lc2,lc3 = st.columns(3)
        with lc1:
            l_date  = st.date_input("Match Date", value=date.today())
            l_team1 = st.selectbox("Team 1", ALL_TEAMS, key="l_t1")
            l_venue = st.selectbox("Venue", ALL_VENUES, key="l_v")
        with lc2:
            l_team2  = st.selectbox("Team 2", [t for t in ALL_TEAMS if t!=l_team1], key="l_t2")
            l_winner = st.selectbox("Winner", [l_team1, l_team2, "No Result"], key="l_w")
            l_pom    = st.selectbox("Player of the Match", [""] + ALL_PLAYERS)
        with lc3:
            l_toss_w  = st.selectbox("Toss Winner", [l_team1, l_team2], key="l_tw")
            l_toss_d  = st.selectbox("Toss Decision", ["field","bat"], key="l_td")
            l_margin  = st.number_input("Margin", min_value=0, value=0)
            l_res_type= st.selectbox("Result Type", ["runs","wickets","super over","No Result"])

        if st.button("✅ Save Match Result", use_container_width=True):
            conn2 = sqlite3.connect(os.path.join(BASE_DIR, "ipl_database.db"))
            conn2.execute("""
                INSERT INTO ipl_2026_live
                (match_date,team1,team2,venue,toss_winner,toss_decision,winner,result_margin,result_type,player_of_match)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (str(l_date), l_team1, l_team2, l_venue, l_toss_w, l_toss_d,
                  l_winner, int(l_margin), l_res_type, l_pom))
            conn2.commit(); conn2.close()
            st.success(f"✅ Match saved: {l_team1} vs {l_team2} — Winner: {l_winner}")
            st.rerun()

    with tab2:
        conn3 = sqlite3.connect(os.path.join(BASE_DIR, "ipl_database.db"))
        live  = pd.read_sql("SELECT * FROM ipl_2026_live ORDER BY match_date DESC", conn3)
        conn3.close()
        if len(live):
            st.subheader("📋 Recent Results")
            st.dataframe(live[["match_date","team1","team2","winner","result_margin","result_type","player_of_match"]],
                         use_container_width=True, hide_index=True)
            st.subheader("🏆 Points Table")
            teams_2026 = pd.concat([live["team1"], live["team2"]]).unique()
            rows = []
            for t in teams_2026:
                tm = live[(live["team1"]==t)|(live["team2"]==t)]
                won  = (tm["winner"]==t).sum()
                lost = len(tm) - won - (tm["winner"]=="No Result").sum()
                rows.append({"Team":t,"P":len(tm),"W":won,"L":lost,"NR":(tm["winner"]=="No Result").sum(),"Pts":won*2})
            pts_df = pd.DataFrame(rows).sort_values("Pts", ascending=False).reset_index(drop=True)
            pts_df.index += 1
            st.dataframe(pts_df, use_container_width=True)
        else:
            st.info("No IPL 2026 match results yet. Add them using the 'Add Match Result' tab!")
            st.markdown("**IPL 2026 season tracking is ready!** Add your first match result above. 🏏")
