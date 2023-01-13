
import pandas as pd    
    

#def data_cleaning(fake_df,real_df):
def data_cleaning(dataframes):
    fake_df = dataframes[0]
    real_df = dataframes[1]
    fake_df.drop(['date', 'subject'], axis=1, inplace=True)
    real_df.drop(['date', 'subject'], axis=1, inplace=True)
    
    ## 0 for fake news, and 1 for real news
    
    fake_df['class'] = 0 
    real_df['class'] = 1

    news_df = pd.concat([fake_df, real_df], ignore_index=True, sort=False)

    news_df['text'] = news_df['title'] + news_df['text']
    news_df.drop('title', axis=1, inplace=True)
    
    return news_df
    