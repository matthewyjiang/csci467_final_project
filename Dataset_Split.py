import random
import csv


def read_file(filename):
    with open(filename, 'r') as file:
        # Using csv.reader since the data might have commas inside the fields
        reader = csv.reader(file)
        data = list(reader)
    return data


def write_to_file(filename, data):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)


def split_data(data):
    total_len = len(data)

    # Determine the lengths of the subsets based on the 1:2:7 ratio
    len_1 = total_len // 10
    len_2 = 2 * len_1

    random.shuffle(data)  # Shuffle the data randomly

    subset_1 = data[:len_1]
    subset_2 = data[len_1:len_1 + len_2]
    subset_3 = data[len_1 + len_2:]

    return subset_1, subset_2, subset_3


def main():
    data = read_file('data/ExtractedTweets_new.csv')

    subset_1, subset_2, subset_3 = split_data(data)

    write_to_file('data/test_NB.csv', subset_1)
    write_to_file('data/dev_NB.csv', subset_2)
    write_to_file('data/train_NB.csv', subset_3)


if __name__ == "__main__":
    main()
