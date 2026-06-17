import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# ============================================================
# LOAD MODEL AND DATA
# ============================================================
model = pickle.load(open('model.pkl', 'rb'))
df = pd.read_csv('results.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.dropna(subset=['home_score', 'away_score'])

def get_result(row):
    if row['home_score'] > row['away_score']:
        return 'home win'
    elif row['home_score'] < row['away_score']:
        return 'away win'
    else:
        return 'draw'

df['result'] = df.apply(get_result, axis=1)

# ============================================================
# GROUPS
# ============================================================
GROUPS = {
    'A': ['Mexico', 'South Africa', 'South Korea', 'Czechia'],
    'B': ['Canada', 'Bosnia and Herzegovina', 'Qatar', 'Switzerland'],
    'C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'],
    'D': ['United States', 'Paraguay', 'Australia', 'Turkey'],
    'E': ['Germany', 'Curacao', 'Ivory Coast', 'Ecuador'],
    'F': ['Netherlands', 'Japan', 'Sweden', 'Tunisia'],
    'G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    'H': ['Spain', 'Cabo Verde', 'Saudi Arabia', 'Uruguay'],
    'I': ['France', 'Senegal', 'Iraq', 'Norway'],
    'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    'K': ['Portugal', 'Congo DR', 'Uzbekistan', 'Colombia'],
    'L': ['England', 'Croatia', 'Ghana', 'Panama'],
}

# Official FIFA Round of 32 structure.
# 'fixed' matches use literal group codes (e.g. winner of A, runner-up of B).
# 'third' matches need the best-ranked qualifying 3rd place team from the listed candidate groups.
ROUND_OF_32_FIXTURES = [
    {"match": "M73", "type": "fixed", "a": ("RU", "A"), "b": ("RU", "B")},
    {"match": "M74", "type": "third", "a": ("W", "E"), "candidates": ["A", "B", "C", "D", "F"]},
    {"match": "M75", "type": "fixed", "a": ("W", "F"), "b": ("RU", "C")},
    {"match": "M76", "type": "fixed", "a": ("W", "C"), "b": ("RU", "F")},
    {"match": "M77", "type": "third", "a": ("W", "I"), "candidates": ["C", "D", "F", "G", "H"]},
    {"match": "M78", "type": "fixed", "a": ("RU", "E"), "b": ("RU", "I")},
    {"match": "M79", "type": "third", "a": ("W", "A"), "candidates": ["C", "E", "F", "H", "I"]},
    {"match": "M80", "type": "third", "a": ("W", "L"), "candidates": ["E", "H", "I", "J", "K"]},
    {"match": "M81", "type": "third", "a": ("W", "D"), "candidates": ["B", "E", "F", "I", "J"]},
    {"match": "M82", "type": "third", "a": ("W", "G"), "candidates": ["A", "E", "H", "I", "J"]},
    {"match": "M83", "type": "fixed", "a": ("RU", "K"), "b": ("RU", "L")},
    {"match": "M84", "type": "fixed", "a": ("W", "H"), "b": ("RU", "J")},
    {"match": "M85", "type": "third", "a": ("W", "B"), "candidates": ["E", "F", "G", "I", "J"]},
    {"match": "M86", "type": "fixed", "a": ("W", "J"), "b": ("RU", "H")},
    {"match": "M87", "type": "third", "a": ("W", "K"), "candidates": ["D", "E", "I", "J", "L"]},
    {"match": "M88", "type": "fixed", "a": ("RU", "D"), "b": ("RU", "G")},
]

# Bracket pairing for Round of 16 onward (by match index in previous round, 0-based)
R16_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)]
QF_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]
SF_PAIRS = [(0, 1), (2, 3)]
FINAL_PAIR = [(0, 1)]

# ============================================================
# HELPER FUNCTIONS (stats + prediction)
# ============================================================
def get_team_matches(data, team):
    matches = data[(data['home_team'] == team) | (data['away_team'] == team)]
    return matches.sort_values('date')

def get_win_rate(data, team, last_n=10):
    matches = get_team_matches(data, team).tail(last_n)
    if len(matches) == 0:
        return 0
    wins = 0
    for _, row in matches.iterrows():
        if row['home_team'] == team and row['result'] == 'home win':
            wins += 1
        elif row['away_team'] == team and row['result'] == 'away win':
            wins += 1
    return wins / len(matches)

def get_avg_goals(data, team, last_n=10):
    matches = get_team_matches(data, team).tail(last_n)
    if len(matches) == 0:
        return {'avg_scored': 0, 'avg_conceded': 0}
    goals_scored, goals_conceded = [], []
    for _, row in matches.iterrows():
        if row['home_team'] == team:
            goals_scored.append(row['home_score'])
            goals_conceded.append(row['away_score'])
        else:
            goals_scored.append(row['away_score'])
            goals_conceded.append(row['home_score'])
    return {
        'avg_scored': sum(goals_scored) / len(goals_scored),
        'avg_conceded': sum(goals_conceded) / len(goals_conceded)
    }

def get_h2h(data, team_a, team_b):
    h2h = data[
        ((data['home_team'] == team_a) & (data['away_team'] == team_b)) |
        ((data['home_team'] == team_b) & (data['away_team'] == team_a))
    ]
    team_a_wins, team_b_wins, draws = 0, 0, 0
    for _, row in h2h.iterrows():
        if row['home_team'] == team_a and row['result'] == 'home win':
            team_a_wins += 1
        elif row['away_team'] == team_a and row['result'] == 'away win':
            team_a_wins += 1
        elif row['result'] == 'draw':
            draws += 1
        else:
            team_b_wins += 1
    total = len(h2h) if len(h2h) > 0 else 1
    return {
        'team_a_h2h_winrate': team_a_wins / total,
        'team_b_h2h_winrate': team_b_wins / total,
    }

@st.cache_data(show_spinner=False)
def predict_match(team_a, team_b):
    a_winrate = get_win_rate(df, team_a)
    b_winrate = get_win_rate(df, team_b)
    a_goals = get_avg_goals(df, team_a)
    b_goals = get_avg_goals(df, team_b)
    h2h = get_h2h(df, team_a, team_b)

    input_features = np.array([[
        a_winrate, b_winrate,
        a_goals['avg_scored'], a_goals['avg_conceded'],
        b_goals['avg_scored'], b_goals['avg_conceded'],
        h2h['team_a_h2h_winrate'], h2h['team_b_h2h_winrate']
    ]])

    prediction = model.predict(input_features)[0]
    probabilities = model.predict_proba(input_features)[0]
    classes = list(model.classes_)

    return {
        'prediction': prediction,
        'home_win_prob': probabilities[classes.index('home win')],
        'draw_prob': probabilities[classes.index('draw')],
        'away_win_prob': probabilities[classes.index('away win')],
    }

def predict_knockout_match(team_a, team_b):
    """Knockout matches can't draw - if model predicts draw, fall back to whichever has higher win prob."""
    result = predict_match(team_a, team_b)
    if result['prediction'] == 'home win':
        return team_a, result
    elif result['prediction'] == 'away win':
        return team_b, result
    else:
        # no draws allowed in knockouts - use win probability to break tie
        if result['home_win_prob'] >= result['away_win_prob']:
            return team_a, result
        else:
            return team_b, result

def get_group_matches(teams):
    matches = []
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            matches.append((teams[i], teams[j]))
    return matches

@st.cache_data(show_spinner=False)
def predict_group_standings(teams_tuple):
    teams = list(teams_tuple)
    points = {team: 0 for team in teams}
    gd = {team: 0 for team in teams}
    gf = {team: 0 for team in teams}
    matches = get_group_matches(teams)
    for team_a, team_b in matches:
        result = predict_match(team_a, team_b)
        pred = result['prediction']
        if pred == 'home win':
            points[team_a] += 3
            gd[team_a] += 1
            gd[team_b] -= 1
            gf[team_a] += 1
        elif pred == 'away win':
            points[team_b] += 3
            gd[team_b] += 1
            gd[team_a] -= 1
            gf[team_b] += 1
        else:
            points[team_a] += 1
            points[team_b] += 1
            gf[team_a] += 1
            gf[team_b] += 1

    standings = pd.DataFrame({
        'Team': list(points.keys()),
        'Pts': list(points.values()),
        'GD': list(gd.values()),
        'GF': list(gf.values()),
    }).sort_values(['Pts', 'GD', 'GF'], ascending=False).reset_index(drop=True)
    standings.index += 1
    return standings, matches


@st.cache_data(show_spinner=False)
def simulate_full_tournament():
    """Run group stage predictions for every group, determine standings,
    rank 3rd place teams, build Round of 32, then simulate through to the Final."""

    all_standings = {}
    winners, runners_up, thirds = {}, {}, {}

    for g, teams in GROUPS.items():
        standings, _ = predict_group_standings(tuple(teams))
        all_standings[g] = standings
        winners[g] = standings.loc[1, 'Team']
        runners_up[g] = standings.loc[2, 'Team']
        thirds[g] = {
            'team': standings.loc[3, 'Team'],
            'Pts': standings.loc[3, 'Pts'],
            'GD': standings.loc[3, 'GD'],
            'GF': standings.loc[3, 'GF'],
            'group': g
        }

    # rank all 12 third-placed teams to find the best 8
    third_list = sorted(
        thirds.values(),
        key=lambda x: (x['Pts'], x['GD'], x['GF']),
        reverse=True
    )
    qualifying_thirds = third_list[:8]
    qualifying_groups = set(t['group'] for t in qualifying_thirds)

    # assign each qualifying third-place team to a Round of 32 "third" slot.
    # this is a constraint-satisfaction problem (not simple greedy) because slot
    # candidate-group lists overlap, so a naive first-match assignment can leave
    # a later slot with no valid team left. Use backtracking to guarantee a
    # complete, valid assignment exists (it always does, by FIFA's design).
    third_slots = [f for f in ROUND_OF_32_FIXTURES if f['type'] == 'third']

    def assign_thirds(slots, available_groups):
        assignment = {}
        used = set()

        def backtrack(i):
            if i == len(slots):
                return True
            slot = slots[i]
            for g in slot['candidates']:
                if g in available_groups and g not in used:
                    used.add(g)
                    assignment[slot['match']] = g
                    if backtrack(i + 1):
                        return True
                    used.remove(g)
                    del assignment[slot['match']]
            return False

        backtrack(0)
        return assignment

    group_assignment = assign_thirds(third_slots, qualifying_groups)
    third_slot_assignment = {
        match: thirds[group]['team'] for match, group in group_assignment.items()
    }

    def resolve_slot(slot):
        kind, group = slot
        if kind == "W":
            return winners[group]
        elif kind == "RU":
            return runners_up[group]

    # build Round of 32 matchups
    r32_matches = []
    for fixture in ROUND_OF_32_FIXTURES:
        team_a = resolve_slot(fixture['a'])
        if fixture['type'] == 'fixed':
            team_b = resolve_slot(fixture['b'])
        else:
            team_b = third_slot_assignment.get(fixture['match'], None)
        r32_matches.append({
            'match': fixture['match'],
            'team_a': team_a,
            'team_b': team_b,
        })

    def simulate_round(matches_list):
        results = []
        for m in matches_list:
            winner, pred = predict_knockout_match(m['team_a'], m['team_b'])
            results.append({**m, 'winner': winner, 'pred': pred})
        return results

    r32_results = simulate_round(r32_matches)

    r16_matches = []
    for i, j in R16_PAIRS:
        r16_matches.append({'team_a': r32_results[i]['winner'], 'team_b': r32_results[j]['winner']})
    r16_results = simulate_round(r16_matches)

    qf_matches = []
    for i, j in QF_PAIRS:
        qf_matches.append({'team_a': r16_results[i]['winner'], 'team_b': r16_results[j]['winner']})
    qf_results = simulate_round(qf_matches)

    sf_matches = []
    for i, j in SF_PAIRS:
        sf_matches.append({'team_a': qf_results[i]['winner'], 'team_b': qf_results[j]['winner']})
    sf_results = simulate_round(sf_matches)

    final_match = [{'team_a': sf_results[0]['winner'], 'team_b': sf_results[1]['winner']}]
    final_results = simulate_round(final_match)

    # third place playoff (losers of semis)
    sf_losers = []
    for res in sf_results:
        loser = res['team_b'] if res['winner'] == res['team_a'] else res['team_a']
        sf_losers.append(loser)
    third_place_match = [{'team_a': sf_losers[0], 'team_b': sf_losers[1]}]
    third_place_results = simulate_round(third_place_match)

    return {
        'group_standings': all_standings,
        'qualifying_thirds': qualifying_thirds,
        'r32': r32_results,
        'r16': r16_results,
        'qf': qf_results,
        'sf': sf_results,
        'final': final_results,
        'third_place': third_place_results,
        'champion': final_results[0]['winner'],
    }


# ============================================================
# UI HELPERS
# ============================================================
def render_match_box(team_a, team_b, date_label="", key_suffix="", clickable=True):
    """Render a small bracket-style box. Returns True if clicked."""
    box_html = f"""
    <div style='
        background-color: #1b1b2a;
        border: 1px solid #3a3a4d;
        border-radius: 8px;
        padding: 8px 10px;
        margin-bottom: 6px;
        font-size: 13px;
        color: white;
    '>
        <div style='display:flex; justify-content:space-between;'>
            <span>{team_a if team_a else 'TBD'}</span>
        </div>
        <div style='border-top: 1px solid #3a3a4d; margin: 4px 0;'></div>
        <div style='display:flex; justify-content:space-between;'>
            <span>{team_b if team_b else 'TBD'}</span>
        </div>
        {f"<div style='color:#888; font-size:11px; margin-top:4px;'>{date_label}</div>" if date_label else ""}
    </div>
    """
    st.markdown(box_html, unsafe_allow_html=True)
    if clickable:
        return st.button("View Prediction", key=f"btn_{key_suffix}", use_container_width=True)
    return False


def show_match_detail(team_a, team_b, knockout=False):
    result = predict_match(team_a, team_b)
    pred = result['prediction']

    if knockout:
        winner, _ = predict_knockout_match(team_a, team_b)
        st.success(f"🏆 Predicted Winner: {winner}")
    else:
        if pred == 'home win':
            st.success(f"🏆 {team_a} wins")
        elif pred == 'away win':
            st.success(f"🏆 {team_b} wins")
        else:
            st.info("🤝 Draw")

    fig = go.Figure(go.Bar(
        x=[f"{team_a} wins", "Draw", f"{team_b} wins"],
        y=[result['home_win_prob'], result['draw_prob'], result['away_win_prob']],
        marker_color=['#2ecc71', '#95a5a6', '#e74c3c'],
        text=[f"{result['home_win_prob']:.0%}", f"{result['draw_prob']:.0%}", f"{result['away_win_prob']:.0%}"],
        textposition='outside'
    ))
    fig.update_layout(
        yaxis_tickformat='.0%',
        yaxis_range=[0, 1],
        height=300,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        margin=dict(t=10, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric(f"{team_a} Win", f"{result['home_win_prob']:.0%}")
    col2.metric("Draw", f"{result['draw_prob']:.0%}")
    col3.metric(f"{team_b} Win", f"{result['away_win_prob']:.0%}")

    # head to head history
    st.markdown("##### Last 5 Head-to-Head Meetings")
    h2h_matches = df[
        ((df['home_team'] == team_a) & (df['away_team'] == team_b)) |
        ((df['home_team'] == team_b) & (df['away_team'] == team_a))
    ].sort_values('date', ascending=False).head(5)

    if len(h2h_matches) == 0:
        st.write("No previous meetings found in the dataset.")
    else:
        for _, match in h2h_matches.iterrows():
            date_str = match['date'].strftime('%d %b %Y')
            home, away = match['home_team'], match['away_team']
            hs, asc = int(match['home_score']), int(match['away_score'])
            tournament = match['tournament']
            if hs > asc:
                w = home
            elif asc > hs:
                w = away
            else:
                w = 'Draw'
            st.markdown(f"**{date_str}** — {tournament}  \n{home} **{hs} - {asc}** {away}  &nbsp; {'🏆 ' + w if w != 'Draw' else '🤝 Draw'}")
            st.divider()


# ============================================================
# PAGE CONFIG + STATE
# ============================================================
st.set_page_config(page_title="World Cup 2026 Predictor", page_icon="⚽", layout="wide")

if 'view' not in st.session_state:
    st.session_state.view = 'home'
if 'selected_group' not in st.session_state:
    st.session_state.selected_group = None
if 'selected_match' not in st.session_state:
    st.session_state.selected_match = None

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    try:
        st.image('logo.png', width=200)
    except Exception:
        pass

st.markdown("""
    <h2 style='text-align: center; color: white; margin-top: 0;'>FIFA World Cup 2026 Predictor</h2>
    <hr>
""", unsafe_allow_html=True)

with st.spinner("Running full tournament simulation..."):
    sim = simulate_full_tournament()

# ============================================================
# MATCH DETAIL VIEW (overlays any page)
# ============================================================
if st.session_state.selected_match is not None:
    team_a, team_b, knockout = st.session_state.selected_match
    if st.button("← Back"):
        st.session_state.selected_match = None
        st.rerun()
    st.markdown(f"## {team_a} vs {team_b}")
    show_match_detail(team_a, team_b, knockout=knockout)

# ============================================================
# GROUP DETAIL VIEW
# ============================================================
elif st.session_state.selected_group is not None:
    group_letter = st.session_state.selected_group
    teams = GROUPS[group_letter]

    if st.button("← Back to Bracket"):
        st.session_state.selected_group = None
        st.rerun()

    st.markdown(f"## Group {group_letter}")
    st.markdown(" | ".join(teams))
    st.write("")

    standings = sim['group_standings'][group_letter]
    matches = get_group_matches(teams)

    st.subheader("📊 Predicted Standings")
    for i, row in standings.iterrows():
        if i <= 2:
            color, label = "#2ecc71", "✅ Advances"
        elif i == 3:
            color, label = "#f39c12", "⚠️ 3rd Place"
        else:
            color, label = "#e74c3c", "❌ Eliminated"
        st.markdown(f"""
        <div style='background-color:#1e1e2e; border-left:4px solid {color}; border-radius:8px;
                    padding:12px 16px; margin-bottom:8px; display:flex; justify-content:space-between;'>
            <span style='color:white; font-size:16px;'><b>{i}.</b> {row['Team']}</span>
            <span style='color:{color};'>{row['Pts']} pts &nbsp; GD {row['GD']:+d} &nbsp; {label}</span>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.subheader("🔮 Match Predictions")
    for team_a, team_b in matches:
        result = predict_match(team_a, team_b)
        pred = result['prediction']
        winner_text = f"🏆 {team_a} wins" if pred == 'home win' else (f"🏆 {team_b} wins" if pred == 'away win' else "🤝 Draw")
        with st.expander(f"{team_a} vs {team_b}   —   {winner_text}"):
            show_match_detail(team_a, team_b)

# ============================================================
# HOME / BRACKET VIEW
# ============================================================
else:
    tab1, tab2 = st.tabs(["🏆 Knockout Bracket", "📋 Group Stage"])

    with tab2:
        st.markdown("### Select a Group")
        group_letters = list(GROUPS.keys())
        for row_start in range(0, 12, 4):
            cols = st.columns(4)
            for col_idx, g in enumerate(group_letters[row_start:row_start + 4]):
                teams = GROUPS[g]
                with cols[col_idx]:
                    st.markdown(f"""
                    <div style='background-color:#1e1e2e; border-radius:12px; padding:14px;
                                margin-bottom:10px; border:1px solid #333;'>
                        <h5 style='color:#f0c040; margin-bottom:8px;'>Group {g}</h5>
                        {''.join([f"<p style='margin:3px 0; color:white; font-size:13px;'>{t}</p>" for t in teams])}
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button("View Group →", key=f"group_{g}"):
                        st.session_state.selected_group = g
                        st.rerun()

    with tab1:
        st.markdown("### Predicted Knockout Bracket")
        st.caption("Click any match to see the win probability breakdown and head-to-head history.")

        # show qualifying 3rd place teams
        with st.expander("📋 Best 8 Third-Placed Teams (click to view ranking)"):
            third_df = pd.DataFrame(sim['qualifying_thirds'])
            third_df = third_df.rename(columns={'team': 'Team', 'group': 'Group'})
            third_df = third_df[['Team', 'Group', 'Pts', 'GD', 'GF']]
            third_df.index = range(1, len(third_df) + 1)
            st.dataframe(third_df, use_container_width=True)

        st.markdown("#### Round of 32")
        cols = st.columns(4)
        for idx, m in enumerate(sim['r32']):
            with cols[idx % 4]:
                clicked = render_match_box(m['team_a'], m['team_b'], date_label=m['match'], key_suffix=f"r32_{idx}")
                if clicked:
                    st.session_state.selected_match = (m['team_a'], m['team_b'], True)
                    st.rerun()

        st.markdown("#### Round of 16")
        cols = st.columns(4)
        for idx, m in enumerate(sim['r16']):
            with cols[idx % 4]:
                clicked = render_match_box(m['team_a'], m['team_b'], key_suffix=f"r16_{idx}")
                if clicked:
                    st.session_state.selected_match = (m['team_a'], m['team_b'], True)
                    st.rerun()

        st.markdown("#### Quarter Finals")
        cols = st.columns(4)
        for idx, m in enumerate(sim['qf']):
            with cols[idx]:
                clicked = render_match_box(m['team_a'], m['team_b'], key_suffix=f"qf_{idx}")
                if clicked:
                    st.session_state.selected_match = (m['team_a'], m['team_b'], True)
                    st.rerun()

        st.markdown("#### Semi Finals")
        cols = st.columns(2)
        for idx, m in enumerate(sim['sf']):
            with cols[idx]:
                clicked = render_match_box(m['team_a'], m['team_b'], key_suffix=f"sf_{idx}")
                if clicked:
                    st.session_state.selected_match = (m['team_a'], m['team_b'], True)
                    st.rerun()

        st.markdown("#### 🥉 Third Place Playoff &nbsp;&nbsp;&nbsp;&nbsp; 🏆 Final")
        cols = st.columns(2)
        with cols[0]:
            m = sim['third_place'][0]
            clicked = render_match_box(m['team_a'], m['team_b'], key_suffix="3rd")
            if clicked:
                st.session_state.selected_match = (m['team_a'], m['team_b'], True)
                st.rerun()
        with cols[1]:
            m = sim['final'][0]
            clicked = render_match_box(m['team_a'], m['team_b'], key_suffix="final")
            if clicked:
                st.session_state.selected_match = (m['team_a'], m['team_b'], True)
                st.rerun()

        st.markdown(f"""
        <div style='text-align:center; margin-top:30px; padding:20px; background-color:#1e1e2e;
                    border-radius:12px; border:2px solid #f0c040;'>
            <h3 style='color:#f0c040;'>🏆 Predicted Champion</h3>
            <h2 style='color:white;'>{sim['champion']}</h2>
        </div>
        """, unsafe_allow_html=True)
