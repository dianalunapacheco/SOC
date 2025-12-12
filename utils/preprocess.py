import pandas as pd
import re

'''
preprocess tweets csv file for because of... dataset
'''

def obtain_file():
    file_tw = pd.read_csv("../data/Because_of_14-30Sept.csv")
    tweets = file_tw.iloc[:, :4]
    return tweets

def clean_data(tweets):
    tweets["Timestamp"] = pd.to_datetime(tweets["Timestamp"], format="%a %b %d %H:%M:%S %z %Y", errors='coerce')
    tweets = tweets.sort_values(by=["Timestamp"])

    if pd.isna(tweets["text"]):
        return ""
    
    tweets["text"] = re.sub(r"\s+", " ", tweets["text"]).strip()
    
    
    return tweets

def split_data():
    rows = []
    print("a")
    

if __name__ == "__main__":
    tweets = obtain_file()
    # print(tweets.head(), "\n", tweets.info(), "\n", tweets.isnull().sum())
    mock_data = clean_data(tweets)
    print(mock_data)
