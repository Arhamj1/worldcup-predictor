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

# page config
st.set_page_config(page_title="World Cup 2026 Predictor", page_icon="⚽")
st.title("⚽ World Cup 2026 Match Predictor")
st.write("Select two teams to predict the match outcome")

# get all teams
all_teams = sorted(list(set(df['home_team'].tolist() + df['away_team'].tolist())))

# team selection
col1, col2 = st.columns(2)
with col1:
    team_a = st.selectbox("Select Team A", all_teams, index=all_teams.index('Brazil'))
with col2:
    team_b = st.selectbox("Select Team B", all_teams, index=all_teams.index('Argentina'))

if st.button("Predict", use_container_width=True):
    if team_a == team_b:
        st.error("Please select two different teams")
    else:
        with st.spinner("Calculating..."):
            a_winrate = get_win_rate(df, team_a)
            b_winrate = get_win_rate(df, team_b)
            a_goals = get_avg_goals(df, team_a)
            b_goals = get_avg_goals(df, team_b)
            h2h = get_h2h(df, team_a, team_b)

            input_features = np.array([[
                a_winrate,
                b_winrate,
                a_goals['avg_scored'],
                a_goals['avg_conceded'],
                b_goals['avg_scored'],
                b_goals['avg_conceded'],
                h2h['team_a_h2h_winrate'],
                h2h['team_b_h2h_winrate']
            ]])

            prediction = model.predict(input_features)[0]
            probabilities = model.predict_proba(input_features)[0]
            classes = model.classes_

            # show prediction
            st.subheader("Prediction")
            if prediction == 'home win':
                st.success(f"🏆 {team_a} wins")
            elif prediction == 'away win':
                st.success(f"🏆 {team_b} wins")
            else:
                st.info("🤝 Draw")

            # probability chart
            fig = go.Figure(go.Bar(
                x=[team_a + ' wins', 'Draw', team_b + ' wins'],
                y=[
                    probabilities[list(classes).index('home win')],
                    probabilities[list(classes).index('draw')],
                    probabilities[list(classes).index('away win')]
                ],
                marker_color=['#1f77b4', '#aec7e8', '#ff7f0e']
            ))
            fig.update_layout(
                title='Win Probabilities',
                yaxis_tickformat='.0%',
                yaxis_range=[0, 1]
            )
            st.plotly_chart(fig, use_container_width=True)

            # team stats
            st.subheader("Team Stats (last 10 matches)")
            stats_col1, stats_col2 = st.columns(2)
            with stats_col1:
                st.metric(f"{team_a} Win Rate", f"{a_winrate:.0%}")
                st.metric(f"{team_a} Avg Goals Scored", f"{a_goals['avg_scored']:.1f}")
                st.metric(f"{team_a} Avg Goals Conceded", f"{a_goals['avg_conceded']:.1f}")
            with stats_col2:
                st.metric(f"{team_b} Win Rate", f"{b_winrate:.0%}")
                st.metric(f"{team_b} Avg Goals Scored", f"{b_goals['avg_scored']:.1f}")
                st.metric(f"{team_b} Avg Goals Conceded", f"{b_goals['avg_conceded']:.1f}")

                # head to head history
            st.subheader(f"Last 5 Head to Head Matches")
            
            h2h_matches = df[
                ((df['home_team'] == team_a) & (df['away_team'] == team_b)) |
                ((df['home_team'] == team_b) & (df['away_team'] == team_a))
            ].sort_values('date', ascending=False).head(5)

            if len(h2h_matches) == 0:
                st.write("No previous meetings found")
            else:
                for _, match in h2h_matches.iterrows():
                    date = match['date'].strftime('%d %b %Y')
                    home = match['home_team']
                    away = match['away_team']
                    hs = int(match['home_score'])
                    as_ = int(match['away_score'])
                    tournament = match['tournament']
                    
                    # figure out winner for color
                    if hs > as_:
                        winner = home
                    elif as_ > hs:
                        winner = away
                    else:
                        winner = 'Draw'
                    
                    st.markdown(f"""
                    **{date}** — {tournament}  
                    {home} **{hs} - {as_}** {away} &nbsp; 
                    {'🏆 ' + winner if winner != 'Draw' else '🤝 Draw'}
                    """)
                    st.divider()