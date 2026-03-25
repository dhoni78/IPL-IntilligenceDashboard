"""
Build a comprehensive player_team_mapping table (player_name, season, team).

Sources:
  1. Deliveries + Matches (2008–2024) — actual match data, highest priority
  2. IPL_Auction CSVs (2013–2026)      — auction assignments
  3. player_team_2026.csv              — current 2026 squad list

The deliveries data uses initials (e.g., "RA Jadeja"), while auction/CSV data
uses full names (e.g., "Ravindra Jadeja"). We keep both naming conventions
since each appears in different parts of the app/DB.
"""

import os
import re
import glob
import sqlite3
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "ipl_database.db")

# ── Team name normalization ──────────────────────────────────────────────────
TEAM_NAME_MAP = {
    # Historical → current franchise names
    "Delhi Daredevils":         "Delhi Capitals",
    "Deccan Chargers":          "Deccan Chargers",       # defunct, keep as-is
    "Kochi Tuskers Kerala":     "Kochi Tuskers Kerala",  # defunct
    "Pune Warriors India":      "Pune Warriors India",   # defunct
    "Rising Pune Supergiant":   "Rising Pune Supergiant",# defunct
    "Rising Pune Supergiants":  "Rising Pune Supergiant",
    "Gujarat Lions":            "Gujarat Lions",         # defunct
    "Kings XI Punjab":          "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "RCB":                      "Royal Challengers Bengaluru",
    "CSK":                      "Chennai Super Kings",
    "MI":                       "Mumbai Indians",
    "KKR":                      "Kolkata Knight Riders",
    "RR":                       "Rajasthan Royals",
    "SRH":                      "Sunrisers Hyderabad",
    "DC":                       "Delhi Capitals",
    "PBKS":                     "Punjab Kings",
    "GT":                       "Gujarat Titans",
    "LSG":                      "Lucknow Super Giants",
}

def normalize_team(name: str) -> str:
    """Normalize team name to current franchise name."""
    if not isinstance(name, str):
        return str(name)
    name = name.strip()
    return TEAM_NAME_MAP.get(name, name)

def clean_player_name(name: str) -> str:
    """Strip whitespace, extra spaces, quotes from player names."""
    if not isinstance(name, str):
        return str(name)
    name = name.strip().strip('"').strip("'")
    name = re.sub(r'\s+', ' ', name)  # collapse multiple spaces
    return name


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 1: Deliveries + Matches (actual match data, 2008–2024)
# ═══════════════════════════════════════════════════════════════════════════════
def extract_from_deliveries(conn: sqlite3.Connection) -> pd.DataFrame:
    """Extract unique (player, team, season) from deliveries joined with matches."""
    print("[1/3] Extracting from deliveries + matches …")

    # Batters
    bat_sql = """
        SELECT DISTINCT d.batter AS player_name,
               d.batting_team AS team,
               m.season
        FROM deliveries d
        JOIN matches m ON d.match_id = m.id
        WHERE d.batter IS NOT NULL AND d.batter != ''
    """
    df_bat = pd.read_sql(bat_sql, conn)

    # Bowlers
    bowl_sql = """
        SELECT DISTINCT d.bowler AS player_name,
               d.bowling_team AS team,
               m.season
        FROM deliveries d
        JOIN matches m ON d.match_id = m.id
        WHERE d.bowler IS NOT NULL AND d.bowler != ''
    """
    df_bowl = pd.read_sql(bowl_sql, conn)

    df = pd.concat([df_bat, df_bowl], ignore_index=True)
    df = df.drop_duplicates(subset=["player_name", "season"])
    df["source"] = "deliveries"
    print(f"   → {len(df)} player-team-season records from deliveries")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 2: Auction CSVs (2013–2026)
# ═══════════════════════════════════════════════════════════════════════════════
def extract_from_auction_csvs() -> pd.DataFrame:
    """Read all IPL_Auction CSVs and extract (player, team, season)."""
    print("[2/3] Extracting from auction CSVs …")
    auction_dir = os.path.join(BASE_DIR, "IPL_Auction")
    rows = []

    for csv_path in sorted(glob.glob(os.path.join(auction_dir, "*.csv"))):
        # Extract year from filename like "IPL_Auction_2022_Sold_Player.csv"
        fname = os.path.basename(csv_path)
        match = re.search(r'(\d{4})', fname)
        if not match:
            continue
        year = int(match.group(1))

        df = pd.read_csv(csv_path)
        # Columns vary, but Name and TeamName are always present
        if "Name" not in df.columns or "TeamName" not in df.columns:
            print(f"   ⚠ Skipping {fname} — missing Name/TeamName columns")
            continue

        for _, row in df.iterrows():
            name = clean_player_name(str(row["Name"]))
            team = normalize_team(str(row["TeamName"]))
            if name and team and name != "nan" and team != "nan":
                rows.append({
                    "player_name": name,
                    "team": team,
                    "season": year,
                })

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["player_name", "season"])
    df["source"] = "auction"
    print(f"   → {len(df)} player-team-season records from auction CSVs")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 3: player_team_2026.csv
# ═══════════════════════════════════════════════════════════════════════════════
def extract_from_2026_csv() -> pd.DataFrame:
    """Read the player_team_2026.csv for 2026 squad assignments."""
    print("[3/3] Extracting from player_team_2026.csv …")
    csv_path = os.path.join(BASE_DIR, "player_team_2026.csv")
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"player_name": "player_name", "team_2026": "team"})
    df["season"] = 2026
    df["player_name"] = df["player_name"].apply(clean_player_name)
    df["team"] = df["team"].apply(normalize_team)
    df = df[["player_name", "team", "season"]].drop_duplicates(subset=["player_name", "season"])
    df["source"] = "squad_2026"
    print(f"   → {len(df)} player-team records from player_team_2026.csv")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# MERGE & VALIDATE
# ═══════════════════════════════════════════════════════════════════════════════
def merge_sources(df_del: pd.DataFrame, df_auc: pd.DataFrame, df_2026: pd.DataFrame) -> pd.DataFrame:
    """Merge all sources with priority: deliveries > auction > squad_2026."""
    print("\nMerging sources (priority: deliveries > auction > squad_2026) …")

    # Combine all
    combined = pd.concat([df_del, df_auc, df_2026], ignore_index=True)

    # Normalize all team names
    combined["team"] = combined["team"].apply(normalize_team)
    combined["player_name"] = combined["player_name"].apply(clean_player_name)

    # For duplicates on (player_name, season), keep highest-priority source
    source_priority = {"deliveries": 0, "auction": 1, "squad_2026": 2}
    combined["priority"] = combined["source"].map(source_priority)
    combined = combined.sort_values("priority")
    combined = combined.drop_duplicates(subset=["player_name", "season"], keep="first")
    combined = combined.drop(columns=["priority", "source"])

    # Sort
    combined = combined.sort_values(["player_name", "season"]).reset_index(drop=True)

    print(f"   → {len(combined)} total unique (player_name, season) records")
    return combined


def validate(df: pd.DataFrame):
    """Run validation checks."""
    print("\nValidation checks:")

    # 1. No nulls
    nulls = df.isnull().sum()
    null_total = nulls.sum()
    print(f"  ✓ Null values: {null_total}" + (" ✗ FAIL" if null_total > 0 else " ✓ PASS"))

    # 2. No duplicate (player_name, season)
    dups = df.duplicated(subset=["player_name", "season"]).sum()
    print(f"  ✓ Duplicate (player, season): {dups}" + (" ✗ FAIL" if dups > 0 else " ✓ PASS"))

    # 3. Season range
    print(f"  ✓ Season range: {df['season'].min()} – {df['season'].max()}")
    print(f"  ✓ Unique players: {df['player_name'].nunique()}")
    print(f"  ✓ Unique teams: {df['team'].nunique()}")
    print(f"  ✓ Seasons covered: {sorted(df['season'].unique())}")

    # 4. Spot-check known transfers
    jadeja = df[df["player_name"].str.contains("Jadeja", case=False, na=False)]
    if len(jadeja):
        print(f"\n  Spot-check — Jadeja's teams:")
        for _, r in jadeja.iterrows():
            print(f"    {r['season']}: {r['team']}")

    return null_total == 0 and dups == 0


def save(df: pd.DataFrame, conn: sqlite3.Connection):
    """Save to DB table and CSV."""
    print("\nSaving …")

    # Drop old table if exists
    conn.execute("DROP TABLE IF EXISTS player_team_mapping")
    conn.execute("""
        CREATE TABLE player_team_mapping (
            player_name TEXT NOT NULL,
            season      INTEGER NOT NULL,
            team        TEXT NOT NULL,
            PRIMARY KEY (player_name, season)
        )
    """)

    # Insert
    df[["player_name", "season", "team"]].to_sql(
        "player_team_mapping", conn, if_exists="append", index=False
    )
    conn.commit()
    print(f"  ✓ Saved {len(df)} rows to player_team_mapping table in DB")

    # CSV export
    csv_path = os.path.join(BASE_DIR, "player_team_mapping.csv")
    df[["player_name", "season", "team"]].to_csv(csv_path, index=False)
    print(f"  ✓ Exported to {csv_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  Building Player-Team Mapping")
    print("=" * 60)

    conn = sqlite3.connect(DB_PATH)

    df_del  = extract_from_deliveries(conn)
    df_auc  = extract_from_auction_csvs()
    df_2026 = extract_from_2026_csv()

    merged = merge_sources(df_del, df_auc, df_2026)

    if validate(merged):
        save(merged, conn)
        print("\n✅ Build complete!")
    else:
        print("\n❌ Validation failed — check warnings above")
        # Save anyway for debugging
        save(merged, conn)
        print("   (saved anyway for inspection)")

    conn.close()
    print("=" * 60)
