import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# load model and data
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

FLAG_EMOJIS = {
    'Mexico': '🇲🇽', 'South Africa': '🇿🇦', 'South Korea': '🇰🇷', 'Czechia': '🇨🇿',
    'Canada': '🇨🇦', 'Bosnia and Herzegovina': '🇧🇦', 'Qatar': '🇶🇦', 'Switzerland': '🇨🇭',
    'Brazil': '🇧🇷', 'Morocco': '🇲🇦', 'Haiti': '🇭🇹', 'Scotland': '🏴󠁧󠁢󠁳󠁣󠁴󠁿',
    'United States': '🇺🇸', 'Paraguay': '🇵🇾', 'Australia': '🇦🇺', 'Turkey': '🇹🇷',
    'Germany': '🇩🇪', 'Curacao': '🇨🇼', 'Ivory Coast': '🇨🇮', 'Ecuador': '🇪🇨',
    'Netherlands': '🇳🇱', 'Japan': '🇯🇵', 'Sweden': '🇸🇪', 'Tunisia': '🇹🇳',
    'Belgium': '🇧🇪', 'Egypt': '🇪🇬', 'Iran': '🇮🇷', 'New Zealand': '🇳🇿',
    'Spain': '🇪🇸', 'Cabo Verde': '🇨🇻', 'Saudi Arabia': '🇸🇦', 'Uruguay': '🇺🇾',
    'France': '🇫🇷', 'Senegal': '🇸🇳', 'Iraq': '🇮🇶', 'Norway': '🇳🇴',
    'Argentina': '🇦🇷', 'Algeria': '🇩🇿', 'Austria': '🇦🇹', 'Jordan': '🇯🇴',
    'Portugal': '🇵🇹', 'Congo DR': '🇨🇩', 'Uzbekistan': '🇺🇿', 'Colombia': '🇨🇴',
    'England': '🏴󠁧󠁢󠁥󠁮󠁧󠁿', 'Croatia': '🇭🇷', 'Ghana': '🇬🇭', 'Panama': '🇵🇦',
}

def flag(team):
    return FLAG_EMOJIS.get(team, '🏳️')

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
    goals_scored = []
    goals_conceded = []
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
    team_a_wins = 0
    team_b_wins = 0
    draws = 0
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

def get_group_matches(teams):
    matches = []
    for i in range(len(teams)):
        for j in range(i+1, len(teams)):
            matches.append((teams[i], teams[j]))
    return matches

def predict_group_standings(teams):
    points = {team: 0 for team in teams}
    gd = {team: 0 for team in teams}
    matches = get_group_matches(teams)
    for team_a, team_b in matches:
        result = predict_match(team_a, team_b)
        pred = result['prediction']
        if pred == 'home win':
            points[team_a] += 3
            gd[team_a] += 1
            gd[team_b] -= 1
        elif pred == 'away win':
            points[team_b] += 3
            gd[team_b] += 1
            gd[team_a] -= 1
        else:
            points[team_a] += 1
            points[team_b] += 1
    standings = pd.DataFrame({
        'Team': list(points.keys()),
        'Pts': list(points.values()),
        'GD': list(gd.values())
    }).sort_values(['Pts', 'GD'], ascending=False).reset_index(drop=True)
    standings.index += 1
    return standings, matches

# page config
st.set_page_config(
    page_title="World Cup 2026 Predictor",
    page_icon="⚽",
    layout="wide"
)

# session state to track which group is selected
if 'selected_group' not in st.session_state:
    st.session_state.selected_group = None

# header
pythoncol1, col2, col3 = st.columns([1, 1, 1])
with col2:
    st.image('logo.jpg', width=250)

st.markdown("""
    <h2 style='text-align: center; color: white; margin-top: 0;'>
        Group Stage Predictor
    </h2>
    <hr>
""", unsafe_allow_html=True)

# if no group selected show homepage
if st.session_state.selected_group is None:
    st.markdown("### 🌍 Select a Group to See Predictions")
    st.write("")

    # show groups in a 4 column grid
    group_letters = list(GROUPS.keys())

    for row_start in range(0, 12, 4):
        cols = st.columns(4)
        for col_idx, group_letter in enumerate(group_letters[row_start:row_start+4]):
            teams = GROUPS[group_letter]
            with cols[col_idx]:
                st.markdown(f"""
                <div style='
                    background-color: #1e1e2e;
                    border-radius: 12px;
                    padding: 16px;
                    margin-bottom: 16px;
                    border: 1px solid #333;
                '>
                    <h4 style='color: #f0c040; margin-bottom: 10px;'>Group {group_letter}</h4>
                    {''.join([f"<p style='margin: 4px 0; color: white;'>{flag(t)} {t}</p>" for t in teams])}
                </div>
                """, unsafe_allow_html=True)

                if st.button(f"View Predictions →", key=f"btn_{group_letter}"):
                    st.session_state.selected_group = group_letter
                    st.rerun()

# group predictions page
else:
    group_letter = st.session_state.selected_group
    teams = GROUPS[group_letter]

    if st.button("← Back to All Groups"):
        st.session_state.selected_group = None
        st.rerun()

    st.markdown(f"## Group {group_letter}")
    st.markdown(" | ".join([f"{flag(t)} {t}" for t in teams]))
    st.write("")

    with st.spinner("Running predictions..."):
        standings, matches = predict_group_standings(teams)

    # standings table with color
    st.subheader("📊 Predicted Standings")

    for i, row in standings.iterrows():
        if i <= 2:
            color = "#2ecc71"  # green - advance
            label = "✅ Advances"
        elif i == 3:
            color = "#f39c12"  # yellow - possible third place
            label = "⚠️ 3rd Place"
        else:
            color = "#e74c3c"  # red - eliminated
            label = "❌ Eliminated"

        st.markdown(f"""
        <div style='
            background-color: #1e1e2e;
            border-left: 4px solid {color};
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
        '>
            <span style='color: white; font-size: 16px;'>
                <b>{i}.</b> {flag(row['Team'])} {row['Team']}
            </span>
            <span style='color: {color};'>
                {row['Pts']} pts &nbsp; GD {row['GD']:+d} &nbsp; {label}
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.subheader("🔮 Match Predictions")

    for team_a, team_b in matches:
        result = predict_match(team_a, team_b)
        pred = result['prediction']

        if pred == 'home win':
            winner_text = f"🏆 {team_a} wins"
        elif pred == 'away win':
            winner_text = f"🏆 {team_b} wins"
        else:
            winner_text = "🤝 Draw"

        with st.expander(f"{flag(team_a)} {team_a} vs {flag(team_b)} {team_b}   —   {winner_text}"):
            fig = go.Figure(go.Bar(
                x=[f"{flag(team_a)} {team_a}", "Draw", f"{flag(team_b)} {team_b}"],
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
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)

            col1, col2, col3 = st.columns(3)
            col1.metric(f"{team_a} Win", f"{result['home_win_prob']:.0%}")
            col2.metric("Draw", f"{result['draw_prob']:.0%}")
            col3.metric(f"{team_b} Win", f"{result['away_win_prob']:.0%}")
