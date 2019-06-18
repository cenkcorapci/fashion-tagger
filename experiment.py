import argparse

import pandas as pd
from tqdm import tqdm

from commons.config import STYLES_DATA_SET_PATH
from commons.data_utils import get_target_list
from experiments.fashion_tagger_experiment import FashionClassifierExperiment
from models.fashion_tagger import FashionTagger

usage_docs = """
--epochs <integer> Number of epochs
--val_split <float> Set validation split(between 0 and 1)
--batch_size <int> Batch size for training
"""

parser = argparse.ArgumentParser(usage=usage_docs)

parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--val_split', type=float, default=.1)
parser.add_argument('--batch_size', type=int, default=128)

args = parser.parse_args()


def get_df():
    df = pd.read_csv(STYLES_DATA_SET_PATH, error_bad_lines=False)
    df = df.dropna()
    df['image'] = df.apply(lambda row: str(row['id']) + ".jpg", axis=1)
    return df


df = get_df()

exclude_list = ['Travel', 'Smart Casual', 'Home', 'Party']
df = df.drop(columns=['productDisplayName', 'year'])

for exclude in exclude_list:
    df = df.loc[df.masterCategory != exclude]

column_list = ['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season']
for column in tqdm(column_list, desc='Pruning attributes by counts'):
    contain_list = []
    for index, row in df.groupby(column).count().iterrows():
        if row['id'] >= 2000:
            contain_list.append(index)
    df = df.loc[df[column].isin(contain_list)]

df = df.drop(columns=['id'])
targets = get_target_list(df)
model = FashionTagger(len(targets))
experiment = FashionClassifierExperiment(df,
                                         targets,
                                         'dense_net_201_fashion_attribute_tagger',
                                         model,
                                         val_split=args.val_split,
                                         nb_epochs=args.epochs,
                                         batch_size=args.batch_size)
experiment.train_model()
