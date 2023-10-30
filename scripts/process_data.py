import pandas as pd

# # Read the data from CSV file
# data = pd.read_csv('../data/ExtractedTweets.csv')

# # Replace Party values
# data['Party'] = data['Party'].replace({'Democrat': 0, 'Republican': 1})

# # create new csv file

# data.to_csv('../data/ExtractedTweets_new.csv', index=False)

data = pd.read_csv('../data/repfulltrain.csv', encoding='utf-8')