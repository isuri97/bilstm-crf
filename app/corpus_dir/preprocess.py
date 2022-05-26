# create a vocabulary form the dataset
# load the dataset
# save the file as vocab.json

import pandas as pd
import csv
import json
import re


def preprocess(df):
    i = 0
    col = []
    for index, row in df.iterrows():
        if '#' in row['TOKEN']:
            col.append(None)
            if '#' not in df.iloc[index + 1]['TOKEN']:
                i += 1
            continue
        else:
            col.append(i)

    df["sent_id"] = col
    df = df[df['sent_id'].notna()]
    df.sent_id = df['sent_id'].astype(int)
    df2 = df[['TOKEN', 'NE-COARSE-LIT', 'sent_id']]
    df2.to_csv('new.txt')

    newlist = []
    with open('readme.txt', 'w', ) as f:
        for i in df2.TOKEN:
            newlist.append(i)
        print(newlist)
    with open('data.json', 'w') as f:
        json.dump(newlist, f)

    ne=pd.read_csv('new.txt')
    m =ne.groupby('sent_id')['TOKEN'].apply(list)
    j = ne.groupby('sent_id')['NE-COARSE-LIT'].apply(list)

    # data= ne[['TOKEN','NE-COARSE-LIT']]
    # df_a=pd.DataFrame(data)
    # df_a.to_csv('we.tsv', sep='\t', index=False)



    # # c=0
    # er = pd.DataFrame()


    df = open('text.txt', 'w')
    for x, y in zip (m, j):
        df.write(str(x).replace('(','').replace(')','')+ '\t' + str(y).replace('(','').replace(')',''))
        df.write('\n')


    df.close()


    # pattern = re.compile(r'\(\w*\)')
    #
    # with open('text.txt', 'r') as f:
    #     input = f.read()
    #     output = re.sub(r'\(\w*\)', '', input)
    #     print(output)


    # for i in df:
    #     re.sub("[\([())\]]", "", i)
    #     print(i)



df = pd.read_csv(
    '/home/isuri/Downloads/bi-lstm-crf-0d4dc92198008e3a38e1247c1adaddb8261ceb99/bi_lstm_crf/app/corpus_dir/topres19th_train.tsv',
    sep='\t', quoting=csv.QUOTE_NONE)


def main():
    preprocess(df)


if __name__ == "__main__":
    main()
# def load_file:
