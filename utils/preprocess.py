import pandas as pd
import re
import os
import kagglehub

'''
preprocessing dataset (tweets) --> because of...
'''

def obtain_file():
    data_path = kagglehub.dataset_download("hossaingh/because-of")
    file_tw = pd.read_csv(os.path.join(data_path, "Because_of_14-30Sept.csv"), low_memory=False)
    tweets = file_tw.iloc[:, :-4] # everything except columns media 1,2,3,4 
    return tweets

def clean_data(tweets):
    emojis = re.compile("["
         u"\U0001F600-\U0001F64F"
         u"\U0001F300-\U0001F5FF"
         u"\U0001F680-\U0001F6FF"
         u"\U0001F1E0-\U0001F1FF"
         "]+", flags=re.UNICODE)

    # delete duplicate data
    tweets = tweets.drop_duplicates(subset=["text"])

    # clean text
    tweets_txt = tweets["text"]
    tweets_txt = str(tweets_txt)
    tweets_txt = re.sub(r"http\S+|www\S+", "", tweets_txt)
    tweets_txt = re.sub(r"@\w+", "", tweets_txt)
    tweets_txt = re.sub(r"#\S+", "", tweets_txt)
    tweets_txt = emojis.sub("", tweets_txt)
    tweets_txt = re.sub(r"[^\w\s]", "", tweets_txt)
    tweets["text"] = tweets_txt

    # tokenizer
    tweets["tokens"] = tweets["text"].apply(lambda x: x.split())

    return tweets

if __name__ == "__main__":
    tweets = obtain_file()
    # print(f"Head: {tweets.head()}\nInfo: {tweets.info()}\n Null Amount: {tweets.isnull().sum()}")
    tweets_copy = tweets.copy()
    mock_data = clean_data(tweets_copy)
    print(mock_data)

    # DELETE THIS COMMENT TO SAVE THE CSV FILE - I COULDNT UPLOAD IT, TOO BIG FOR GITHUB
    # mock_data.to_csv("../data/clean_data.csv", index=False)
