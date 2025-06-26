import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
hist_df = pd.read_csv('historical_data.csv')
fg_df = pd.read_csv('fear_greed_index.csv')

# Convert dates
hist_df['Timestamp IST'] = pd.to_datetime(hist_df['Timestamp IST'], format='%d-%m-%Y %H:%M')
fg_df['date'] = pd.to_datetime(fg_df['date'])
hist_df['date'] = pd.to_datetime(hist_df['Timestamp IST'].dt.date)

# Add formatted date
hist_df['formatted_date'] = hist_df['date'].dt.strftime('%d-%m-%Y')
fg_df['formatted_date'] = fg_df['date'].dt.strftime('%d-%m-%Y')

# Merge
merged_df = hist_df.merge(fg_df[['date', 'classification']], on='date', how='left')
merged_df['Closed PnL'] = merged_df['Closed PnL'].fillna(0)
merged_df['Size USD'] = merged_df['Size USD'].replace(0, np.nan)

# Trader performance metrics
performance = merged_df.groupby(['Account', 'classification']).agg(
    total_trades=('Trade ID', 'count'),
    win_rate=('Closed PnL', lambda x: (x > 0).sum() / len(x)),
    total_pnl=('Closed PnL', 'sum'),
    avg_pnl=('Closed PnL', 'mean'),
    total_usd=('Size USD', 'sum'),
    pnl_std=('Closed PnL', 'std')
).reset_index()

performance['ROI'] = performance['total_pnl'] / performance['total_usd']
performance['sharpe_proxy'] = performance['ROI'] / performance['pnl_std']

# Streamlit App
st.set_page_config(page_title="Trader Performance Dashboard", layout="wide")

st.title("📊 Trader Performance vs Market Sentiment")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Visualizations", "Top Traders", "Advanced Insights"])

with tab1:
    st.header("📁 Raw Data Preview")
    st.write("Historical Trade Data with Market Sentiment:")
    st.dataframe(merged_df.head(50))

    st.header("🧮 Trader Performance Metrics")
    st.dataframe(performance.head(50))

with tab2:
    st.subheader("1️⃣ ROI Distribution by Sentiment")
    fig1, ax1 = plt.subplots()
    sns.boxplot(x='classification', y='ROI', data=performance, ax=ax1)
    st.pyplot(fig1)

    st.subheader("2️⃣ Trade Volume by Sentiment")
    trade_counts = merged_df.groupby('classification')['Trade ID'].count().reset_index()
    fig2, ax2 = plt.subplots()
    sns.barplot(x='classification', y='Trade ID', data=trade_counts, ax=ax2)
    st.pyplot(fig2)

    st.subheader("3️⃣ Win Rate by Sentiment")
    win_rate_sentiment = merged_df.groupby('classification').apply(
        lambda x: (x['Closed PnL'] > 0).sum() / len(x)
    ).reset_index(name='win_rate')
    fig3, ax3 = plt.subplots()
    sns.barplot(x='classification', y='win_rate', data=win_rate_sentiment, ax=ax3)
    st.pyplot(fig3)

    st.subheader("4️⃣ PnL Distribution by Sentiment")
    fig4, ax4 = plt.subplots()
    sns.violinplot(x='classification', y='Closed PnL', data=merged_df, ax=ax4)
    st.pyplot(fig4)

    st.subheader("5️⃣ Heatmap: ROI of Traders Across Sentiments")
    pivot = performance.pivot(index='Account', columns='classification', values='ROI')
    fig5, ax5 = plt.subplots(figsize=(11, 8))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax5)
    st.pyplot(fig5)

with tab3:
    st.header("📈 Top Traders")

    for sentiment in performance['classification'].unique():
        st.subheader(f"Top 5 Traders during {sentiment}")
        filtered = performance[performance['classification'] == sentiment]
        top5 = filtered.sort_values(by='ROI', ascending=False).head(5)
        st.dataframe(top5[['Account', 'ROI', 'win_rate', 'avg_pnl']])

with tab4:
    st.header("🔬 Advanced Insights")

    st.subheader("Consistently Profitable Traders (ROI > 20% in all sentiment regimes)")
    roi_consistency = performance.pivot(index='Account', columns='classification', values='ROI').dropna()
    consistent = roi_consistency[(roi_consistency > 0.2).all(axis=1)]
    st.dataframe(consistent)

    st.subheader("Top Sharpe Proxy Traders")
    top_sharpe = performance.sort_values('sharpe_proxy', ascending=False).head(10)
    st.dataframe(top_sharpe[['Account', 'classification', 'ROI', 'pnl_std', 'sharpe_proxy']])
