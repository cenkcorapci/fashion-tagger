import argparse

import pandas as pd
from tqdm import tqdm

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


# Category Classifier ---------------------------------------------------------
df = get_df()


def category_extractor(row):
    if row['subCategory'] in ['Topwear', 'Dress', 'Headwear', 'Innerwear']:
        if len(df.loc[df.articleType == str(row['articleType'])]) >= 800:
            return str(row['articleType'])
    if row['subCategory'] in ['Bottomwear', 'Shoes'] and len(df.loc[df.articleType == str(row['articleType'])]) >= 300:
        return str(row['articleType'])
    return str(row['subCategory'])


exclude_list = ['Vouchers', 'Home Furnishing', 'Umbrellas', 'Water Bottle', 'Bath and Body',
                'Shoe Accessories', 'Sports Accessories', 'Sports Equipment', 'Free Gifts',
                'Cufflinks', 'Apparel Set', 'Cufflinks', 'Green', 'Wristbands']
tqdm.pandas()

df['subCategory'] = df.progress_apply(lambda row: category_extractor(row), axis=1)

for exclude in exclude_list:
    df = df.loc[df.subCategory != exclude]

df.subCategory = df.subCategory.replace({'Perfumes': 'Green',
                                         'Lips': 'Cosmetic',
                                         'Eyes': 'Cosmetic',
                                         'Sandals': 'Sandal',
                                         'Skin Care': 'Cosmetic',
                                         'Makeup': 'Cosmetic',
                                         'Skin': 'Cosmetic',
                                         'Hair': 'Cosmetic',
                                         'Nails': 'Cosmetic',
                                         'Beauty Accessories': 'Cosmetic',
                                         'Mufflers': 'Scarves',
                                         'Stoles': 'Scarves'
                                         })

df = df[['image', 'subCategory']]
df.columns = ['image', 'target']

model = FashionTagger(len(df.target.unique()))
experiment = FashionTaggerExperiment(df,
                                     'mobile_net_v2_fashion_category',
                                     model,
                                     val_split=args.val_split,
                                     nb_epochs=args.epochs,
                                     batch_size=args.batch_size)
experiment.train_model()

# Color Classifier ---------------------------------------------------------
df = get_df()
df.baseColour = df.baseColour.replace({'Lime Green': 'Green',
                                       'Fluorescent Green': 'Green',
                                       'Sea Green': 'Green',
                                       'Mushroom Brown': 'Brown',
                                       'Coffee Brown': 'Brown',
                                       'Bronze': 'Brown',
                                       'Copper': 'Brown',
                                       'Rose': 'Red',
                                       'Burgundy': 'Purple',
                                       'Metallic': 'Grey',
                                       'Mustard': 'Yellow',
                                       'Nude': 'Beige',
                                       'Taupe': 'Grey',
                                       'Mauve': 'Pink',
                                       'Turquoise Blue': 'Teal',
                                       'Maroon': 'Red',
                                       'Rust': 'Orange',
                                       'Skin': 'Beige',
                                       'Tan': 'Beige',
                                       'Off White': 'White'})
df = df[['image', 'baseColour']]
df.columns = ['image', 'target']

model = FashionTagger(len(df.target.unique()))
experiment = FashionTaggerExperiment(df,
                                     'mobile_net_v2_fashion_base_color',
                                     model,
                                     val_split=args.val_split,
                                     nb_epochs=args.epochs,
                                     batch_size=args.batch_size)
experiment.train_model()

# season classifier ---------------------------------------------------------
df = get_df()

df = df[['image', 'season']]
df.columns = ['image', 'target']
model = FashionTagger(len(df.target.unique()))
experiment = FashionTaggerExperiment(df,
                                     'mobile_net_v2_fashion_season',
                                     model,
                                     val_split=args.val_split,
                                     nb_epochs=args.epochs,
                                     batch_size=args.batch_size)

# gender classifier ---------------------------------------------------------
df = get_df()
df.gender = df.gender.replace({'Boys': 'Men', 'Girls': 'Women'})
df = df[['image', 'gender']]
df.columns = ['image', 'target']
model = FashionTagger(len(df.target.unique()))
experiment = FashionTaggerExperiment(df,
                                     'mobile_net_v2_fashion_gender',
                                     model,
                                     val_split=args.val_split,
                                     nb_epochs=args.epochs,
                                     batch_size=args.batch_size)
experiment.train_model()

# usage classifier ---------------------------------------------------------
df = get_df()
df.usage = df.usage.replace({'Smart Casual': 'Casual'})
df.usage = df.usage.replace({'Home': 'Other', 'Travel': 'Other', 'Party': 'Other'})
df = df.loc[df.usage != 'Other']

df = df[['image', 'usage']]
df.columns = ['image', 'target']
model = FashionTagger(len(df.target.unique()))
experiment = FashionTaggerExperiment(df,
                                     'mobile_net_v2_fashion_usage',
                                     model,
                                     val_split=args.val_split,
                                     nb_epochs=args.epochs,
                                     batch_size=args.batch_size)
experiment.train_model()
