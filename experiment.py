import argparse

import pandas as pd

from commons.config import STYLES_DATA_SET_PATH
from experiments.fashion_tagger_experiment import FashionTaggerExperiment
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


# usage classifier
df = get_df()
df.usage = df.usage.replace({'Smart Casual': 'Casual'})
df.usage = df.usage.replace({'Home': 'Other', 'Travel': 'Other', 'Party': 'Other'})
df = df[['image', 'usage']]
df.columns = ['image', 'target']
model = FashionTagger(len(df.target.unique()))
experiment = FashionTaggerExperiment(df,
                                     'Xception_fashion_usage',
                                     model,
                                     val_split=args.val_split,
                                     nb_epochs=args.epochs,
                                     batch_size=args.batch_size)
experiment.train_model()
