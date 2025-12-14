import pandas as pd
import re
import os
import kagglehub
import warnings

warnings.filterwarnings('ignore')


# CONFIGURATION
class Config:
    """Configuration class for model and analysis parameters."""
    # determine the run mode, testing, local or cloud
    RUN_MODE = "CLOUD"

    # ParsBERT Model
    MODEL_NAME = "HooshvareLab/bert-fa-base-uncased"
    MAX_LENGTH = 128
    LEARNING_RATE = 2e-5
    WARMUP_STEPS = 500
    WEIGHT_DECAY = 0.01

    # emotion labels (Ekman's 6 + Neutral)
    LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
    LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
    ID2LABEL = {idx: label for idx, label in enumerate(LABELS)}

    # high-arousal emotions for intensity calculation (Used by SNA)
    HIGH_AROUSAL = ["anger", "fear", "disgust"]

    # time periods (Used by SNA for temporal analysis)
    TIME_PERIODS = {
        'Period 1': ('2022-09-14', '2022-09-30'),
        'Period 2': ('2022-10-01', '2022-10-31'),
        'Period 3': ('2022-11-01', '2022-11-30'),
        'Period 4': ('2022-12-01', '2022-12-31'),
        'Period 5': ('2023-01-01', '2023-01-31'),
    }

    # bot filtering thresholds (Used by SNA)
    BOT_TWEET_THRESHOLD = 50
    BOT_FOLLOWER_RATIO_LOW = 0.1
    BOT_FOLLOWER_RATIO_HIGH = 10
    BOT_CREATION_DATE = '2022-09-16'

    # dynamic parameters
    if RUN_MODE == "TEST":
        # for fast debugging
        TRAIN_SAMPLES = 50
        NUM_EPOCHS = 1
        BATCH_SIZE = 8
    elif RUN_MODE == "LOCAL":
        TRAIN_SAMPLES = 2000
        NUM_EPOCHS = 3
        BATCH_SIZE = 8
    elif RUN_MODE == "CLOUD":
        TRAIN_SAMPLES = None
        NUM_EPOCHS = 4
        BATCH_SIZE = 32


# get data
def obtain_file_a():
    """Download the 'Because of...' Persian dataset (Dataset A)"""
    print("Downloading Dataset A (Persian tweets) from Kaggle...")
    data_path = kagglehub.dataset_download("hossaingh/because-of")
    file_tw = pd.read_csv(os.path.join(data_path,
                                       "Because_of_14-30Sept.csv"),
                          low_memory=False)
    tweets = file_tw.iloc[:, :-4]
    return tweets


def obtain_file_b():
    """Download the 'Iran Protests' English dataset (Dataset B)"""
    print("Downloading Dataset B (English tweets) from Kaggle...")
    data_path_b = kagglehub.dataset_download(
        "konradb/iran-protests-2022-tweets")
    csv_files = [f for f in os.listdir(data_path_b) if f.endswith('.csv')]
    if csv_files:
        tweets = pd.read_csv(os.path.join(data_path_b, csv_files[0]),
                             low_memory=False)
        return tweets
    raise FileNotFoundError("No CSV file found in Dataset B download.")


# clean data
def clean_data_a(tweets):
    """Clean and preprocess Persian tweet data (Dataset A)"""
    print("Cleaning Dataset A...")

    # emoji pattern
    emojis = re.compile(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        "]+", flags=re.UNICODE)

    tweets = tweets.drop_duplicates(subset=["text"])

    def clean_single_text(text):
        if pd.isna(text):
            return ""
        text = str(text)
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#\S+", "", text)
        text = emojis.sub("", text)
        # keep Persian chars + alphanumeric
        text = re.sub(r"[^\w\s\u0600-\u06FF]", "", text)
        return text.strip()

    tweets["text"] = tweets["text"].apply(clean_single_text)
    tweets = tweets[tweets["text"].str.len() > 0]
    tweets["tokens"] = tweets["text"].apply(lambda x: x.split())

    print(f"Cleaned {len(tweets)} tweets in Dataset A.")
    return tweets


# bot filtering for dataset B
def filter_bots_b(tweets_df):
    """
    Filter out potential bot accounts from Dataset B based on criteria.
    """
    print("\nFiltering potential bot accounts from Dataset B...")
    initial_count = len(tweets_df)

    if 'user_created' in tweets_df.columns:
        tweets_df['user_created'] = pd.to_datetime(tweets_df['user_created'],
                                                   errors='coerce')

    if 'date' in tweets_df.columns:
        tweets_df['date'] = pd.to_datetime(tweets_df['date'], errors='coerce')

    if 'user_created' in tweets_df.columns:
        tweets_df['flag_new_account'] = tweets_df['user_created'] > \
            pd.to_datetime(Config.BOT_CREATION_DATE)
    else:
        tweets_df['flag_new_account'] = False

    if 'user_tweets' in tweets_df.columns and \
       'user_created' in tweets_df.columns:
        tweets_df['account_age_days'] = \
            (pd.Timestamp.now() - tweets_df['user_created']).dt.days
        tweets_df['account_age_days'] = \
            tweets_df['account_age_days'].clip(lower=1)
        tweets_df['tweets_per_day'] = \
            tweets_df['user_tweets'] / tweets_df['account_age_days']
        tweets_df['flag_high_frequency'] = \
            tweets_df['tweets_per_day'] > Config.BOT_TWEET_THRESHOLD
    else:
        tweets_df['flag_high_frequency'] = False

    if 'user_followers' in tweets_df.columns and \
       'user_following' in tweets_df.columns:
        tweets_df['follower_following_ratio'] = \
            tweets_df['user_followers'] / (tweets_df['user_following'] + 1)
        tweets_df['flag_suspicious_ratio'] = (
            (tweets_df['follower_following_ratio'] <
             Config.BOT_FOLLOWER_RATIO_LOW) |
            (tweets_df['follower_following_ratio'] >
             Config.BOT_FOLLOWER_RATIO_HIGH)
        )
    else:
        tweets_df['flag_suspicious_ratio'] = False

    tweets_df['total_flags'] = (
        tweets_df['flag_new_account'].astype(int) +
        tweets_df['flag_high_frequency'].astype(int) +
        tweets_df['flag_suspicious_ratio'].astype(int)
    )

    # filter out accounts with 2 or more flags
    filtered_df = tweets_df[tweets_df['total_flags'] < 2].copy()

    removed = initial_count - len(filtered_df)
    print(f"Removed {removed} potential bot accounts "
          f"({removed/initial_count*100:.1f}%)")
    print(f"Remaining tweets in Dataset B: {len(filtered_df)}")

    return filtered_df
