import pandas as pd
import os
import networkx as nx
import matplotlib.pyplot as plt
# Import Config and functions from the preprocessing file
from preprocessing import (
    Config, obtain_file_b, filter_bots_b
)


# calculate emotional intensity
def calculate_emotional_intensity(tweets_df, date_column='date'):
    """
    Calculate emotional intensity (proportion of high-arousal emotions)
    for Period 1 (where emotion data exists).
    """
    print("\n" + "="*60)
    print("Calculating emtotional intensity (Period 1 Only)")
    print("="*60)

    tweets_df.rename(columns={"Timestamp": "date"}, inplace=True)
    tweets_df[date_column] = pd.to_datetime(tweets_df[date_column])
    tweets_df['is_high_arousal'] = \
        tweets_df['predicted_emotion'].isin(Config.HIGH_AROUSAL)

    period_intensity = {}
    period_name = 'Period 1'
    start, end = Config.TIME_PERIODS[period_name]
    mask = (tweets_df[date_column] >= start) & (tweets_df[date_column] <= end)
    period_tweets = tweets_df[mask]

    if len(period_tweets) > 0:
        high_arousal = period_tweets['is_high_arousal'].sum()
        total = len(period_tweets)
        intensity = high_arousal / total

        period_intensity[period_name] = {
            'start': start,
            'end': end,
            'total_tweets': total,
            'high_arousal_count': high_arousal,
            'emotional_intensity': intensity
        }

    print("\nEmotional Intensity by Time Period:")
    for period, data in period_intensity.items():
        print(f"{period}: {data['emotional_intensity']:.3f} "
              f"({data['high_arousal_count']}/{data['total_tweets']} tweets)")

    period_df = pd.DataFrame.from_dict(period_intensity, orient='index')
    period_df.to_csv('../data/emotional_intensity_by_period.csv')

    return period_intensity


# SNA analysis
def build_retweet_network(tweets_df, period_name, start_date, end_date):
    """
    Build retweet network for a specific time period. (Variable B)
    """
    tweets_df['date'] = pd.to_datetime(tweets_df['date'])
    mask = (tweets_df['date'] >= start_date) & \
           (tweets_df['date'] <= end_date)
    period_tweets = tweets_df[mask]

    print(f"\n{period_name}: {len(period_tweets)} tweets")

    G = nx.DiGraph()

    if 'user_id' in period_tweets.columns and \
       'retweeted_user_id' in period_tweets.columns:
        for _, row in period_tweets.iterrows():
            if pd.notna(row['retweeted_user_id']):
                user = row['user_id']
                retweeted = row['retweeted_user_id']

                if G.has_edge(user, retweeted):
                    G[user][retweeted]['weight'] += 1
                else:
                    G.add_edge(user, retweeted, weight=1)

    return G, period_tweets


def analyze_network_structure(G, period_name):
    """
    Analyze network structure: modularity, centrality measures.
    """
    print(f"\nAnalyzing network structure for {period_name}...")

    metrics = {
        'period': period_name,
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
    }

    if G.number_of_nodes() > 0:
        try:
            communities = nx.community.greedy_modularity_communities(
                G.to_undirected())
            modularity = nx.community.modularity(G.to_undirected(),
                                                 communities)
            metrics['modularity'] = modularity
            metrics['num_communities'] = len(communities)
            print(f"Modularity: {modularity:.3f}")
        except Exception:
            metrics['modularity'] = None
            metrics['num_communities'] = None

        try:
            eigenvector_cent = nx.eigenvector_centrality(G, max_iter=1000)
            betweenness_cent = nx.betweenness_centrality(G)

            top_eigen = sorted(eigenvector_cent.items(),
                               key=lambda x: x[1], reverse=True)[:10]
            top_between = sorted(betweenness_cent.items(),
                                 key=lambda x: x[1], reverse=True)[:10]

            metrics['top_eigenvector_centrality'] = top_eigen
            metrics['top_betweenness_centrality'] = top_between
        except Exception:
            metrics['top_eigenvector_centrality'] = []
            metrics['top_betweenness_centrality'] = []

    return metrics


def visualize_temporal_correlation_period1(correlation_df):
    """
    Visualize temporal patterns of network metrics (all periods) and
    emotional baseline (Period 1).
    """
    print("\nGenerating visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    periods = correlation_df['period'].tolist()

    # plot 1: network modularity evolution (all periods)
    if 'modularity' in correlation_df.columns:
        ax1 = axes[0, 0]
        modularity = correlation_df['modularity'].tolist()
        colors = ['red' if p == 'Period 1' else 'blue' for p in periods]
        ax1.bar(periods, modularity, color=colors, alpha=0.7)
        ax1.set_xlabel('Time Period', fontsize=12)
        ax1.set_ylabel('Modularity (Polarization)', fontsize=12)
        ax1.set_title('Network Polarization Across Time Periods', fontsize=14,
                      fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.legend(['Period 1 (with emotion data)', 'Other periods'],
                   loc='upper right')

    # plot 2: network size evolution (all periods)
    if 'nodes' in correlation_df.columns:
        ax2 = axes[0, 1]
        nodes = correlation_df['nodes'].tolist()
        ax2.plot(periods, nodes, marker='o', linewidth=2, markersize=8,
                 color='green')
        ax2.scatter([periods[0]], [nodes[0]], s=200, c='red', zorder=5,
                    label='Period 1 (with emotion data)')
        ax2.set_xlabel('Time Period', fontsize=12)
        ax2.set_ylabel('Number of Nodes', fontsize=12)
        ax2.set_title('Network Size Evolution', fontsize=14,
                      fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

    # plot 3: period 1 deep dive (emotion + network)
    ax3 = axes[1, 0]
    period1_data = correlation_df[correlation_df['period'] ==
                                  'Period 1'].iloc[0]
    metrics = []
    values = []

    if pd.notna(period1_data.get('emotional_intensity')):
        metrics.append('Emotional\nIntensity')
        values.append(period1_data['emotional_intensity'])

    if pd.notna(period1_data.get('modularity')):
        metrics.append('Network\nModularity')
        values.append(period1_data['modularity'])

    if len(metrics) > 0:
        colors_bar = ['red', 'blue'][:len(metrics)]
        ax3.bar(metrics, values, color=colors_bar, alpha=0.7)
        ax3.set_ylabel('Score', fontsize=12)
        ax3.set_title('Period 1: Emotion-Network Baseline', fontsize=14,
                      fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_ylim(0, 1)

    # plot 4 : research design summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = """
RESEARCH DESIGN SUMMARY
=======================================
Emotion Data (A): Period 1 Baseline
Network Data (B): Periods 1-5 Evolution

Interpretation: Measures how the initial emotional
surge (Period 1) relates to the long-term changes
in network polarization and size (Periods 1-5).
"""

    ax4.text(0.1, 0.9, summary_text, fontsize=10, verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig('./outputs/correlation_plots.png', dpi=300,
                bbox_inches='tight')
    print("Saved visualization to './outputs/correlation_plots.png'")
    plt.close()


# main pipeline
def run_network_analysis():
    """Run the complete network analysis pipeline."""
    print("\n" + "="*70)
    print(" MAHSA AMINI MOVEMENT: NETWORK DYNAMICS ANALYSIS (SNA) ")
    print("="*70)

    os.makedirs('../data', exist_ok=True)
    os.makedirs('./outputs', exist_ok=True)

    # load emotion data (from previous NLP step)
    emotion_file = '../data/tweets_with_emotions.csv'
    if not os.path.exists(emotion_file):
        print(f"\nERROR: Emotion data missing at {emotion_file}")
        print("Please run 'sentiment_analysis.py' first.")
        return
    tweets_a_emotions = pd.read_csv(emotion_file)
    print(f"\nLoaded {len(tweets_a_emotions)} tweets with emotions.")

    # calculate emotional intensity (Variable A)
    period_intensity = calculate_emotional_intensity(tweets_a_emotions)

    # load and filter Dataset B for SNA
    print("\nLOADING DATASET B FOR NETWORK ANALYSIS")
    sna_file = '../data/tweets_filtered_for_sna.csv'

    if os.path.exists(sna_file):
        tweets_b = pd.read_csv(sna_file)
        print(f"Loaded existing Dataset B: {len(tweets_b)} tweets")
    else:
        try:
            tweets_b = obtain_file_b()
            tweets_b = filter_bots_b(tweets_b)
            tweets_b.to_csv(sna_file, index=False)
            print(f"Saved filtered Dataset B to '{sna_file}'")
        except Exception as e:
            print(f"Warning: Could not download/process Dataset B: {e}")
            print("Please check internet connection or manually download.")
            tweets_b = None

    # build and analyze retweet networks over time periods
    if tweets_b is not None:
        print("\nNETWORK ANALYSIS BY TIME PERIOD")
        network_metrics = []

        for period_name, (start, end) in Config.TIME_PERIODS.items():
            G, period_tweets = build_retweet_network(tweets_b,
                                                     period_name, start, end)
            metrics = analyze_network_structure(G, period_name)
            network_metrics.append(metrics)

        network_df = pd.DataFrame(network_metrics)
        network_df.to_csv('../data/network_metrics_by_period.csv',
                          index=False)
        print("\nSaved network metrics to '../data/"
              "network_metrics_by_period.csv'")
    else:
        print("\nSKIPPED - Dataset B not available")
        return

    # temporal correlation analysis
    print("\nTEMPORAL CORRELATION ANALYSIS")
    network_df['emotional_intensity'] = network_df['period'].map({
        'Period 1': period_intensity.get('Period 1', {}).get(
            'emotional_intensity'),
    })

    correlation_analysis = network_df.copy()
    correlation_analysis.to_csv(
        '../data/temporal_correlation_analysis.csv', index=False)
    print("\nSaved correlation analysis to '../data/"
          "temporal_correlation_analysis.csv'")

    # visualisation
    visualize_temporal_correlation_period1(correlation_analysis)

    print("\n" + "="*70)
    print("SNA complete ")
    print("="*70)
    print("Final output: ../data/temporal_correlation_analysis.csv")


if __name__ == "__main__":
    run_network_analysis()
