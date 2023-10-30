import pandas as pd
#read txt file



# # Read the data from CSV file
# data = pd.read_csv('../data/ExtractedTweets.csv')

# # Replace Party values
# data['Party'] = data['Party'].replace({'Democrat': 0, 'Republican': 1})

# # create new csv file

# data.to_csv('../data/ExtractedTweets_new.csv', index=False)

train_list = []

convert_dict = {'Tweet': str,
                'Party': int
                }

with open("../data/repfulltrain.txt") as f:
    for line in f:
        if(len(line.strip()) > 0):
            train_list.append([1, line.strip()])
        
with open("../data/demfulltrain.txt") as f:
    for line in f:
        if(len(line.strip()) > 0):
            train_list.append([0, line.strip()])
        
data = pd.DataFrame(train_list, columns=['Party', 'Tweet'])
data = data.astype(convert_dict)

data.to_csv('../data/train_set.csv', index=False)


test_list = []

with open("../data/repfullval.txt") as f:
    for line in f:
        if(len(line.strip()) > 0):
            test_list.append([1, line.strip()])
        
with open("../data/demfullval.txt") as f:
    for line in f:
        if(len(line.strip()) > 0):
            test_list.append([0, line.strip()])
        
test_data = pd.DataFrame(test_list, columns=['Party', 'Tweet'])
test_data = test_data.astype(convert_dict)
        
test_data.to_csv('../data/test_set.csv', index=False)

    