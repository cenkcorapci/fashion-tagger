import argparse
import logging

import pandas as pd
from tqdm import tqdm

from commons.config import STYLES_DATA_SET_PATH
from experiments.fashion_tagger_experiment import FashionTaggerExperiment
from models.fashion_tagger import FashionTagger, FashionTaggerModels

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


# Master category classifier
try:
    df = get_df()

    exclude_list = ['Free Items', 'Sporting Goods', 'Home']

    for exclude in exclude_list:
        df = df.loc[df.masterCategory != exclude]

    df = df[['image', 'masterCategory']]
    df.columns = ['image', 'target']
    model = FashionTagger(len(df.target.unique()), FashionTaggerModels.dense201)
    experiment = FashionTaggerExperiment(df,
                                         'dense_net_201_fashion_master_category',
                                         model,
                                         val_split=args.val_split,
                                         nb_epochs=args.epochs,
                                         batch_size=args.batch_size)
    experiment.train_model()

except Exception as exp:
    print("Can not train a season classifier", exp)

# usage classifier ---------------------------------------------------------
try:
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

except Exception as exp:
    logging.error("Can not train a usage classifier", exp)

# season classifier ---------------------------------------------------------
try:
    df = get_df()
    df = df.loc[df.masterCategory.isin(['Footwear', 'Apparel'])]

    df = df[['image', 'season']]
    df.columns = ['image', 'target']
    model = FashionTagger(len(df.target.unique()))
    experiment = FashionTaggerExperiment(df,
                                         'mobile_net_v2_fashion_season',
                                         model,
                                         val_split=args.val_split,
                                         nb_epochs=args.epochs,
                                         batch_size=args.batch_size)
    experiment.train_model()

except Exception as exp:
    print("Can not train a season classifier", exp)

# Color Classifier ---------------------------------------------------------
try:

    def color_extractor(row):
        if len(df.loc[df.baseColour == str(row['baseColour'])]) >= 500:
            return str(row['baseColour'])
        else:
            return 'Other'


    df = get_df()
    df = df.loc[df.masterCategory.isin(['Footwear', 'Apparel'])]

    tqdm.pandas()
    df['baseColour'] = df.progress_apply(lambda row: color_extractor(row), axis=1)
    df = df.loc[df.baseColour != 'Other']

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
except Exception as exp:
    logging.error("Can not train a base color classifier", exp)

# Category Classifier ---------------------------------------------------------
try:
    df = get_df()


    def category_extractor(row):
        if row['masterCategory'] == 'Personal Care':
            return 'Personal Care'
        if row['subCategory'] in ['Topwear', 'Dress', 'Headwear', 'Innerwear']:
            if len(df.loc[df.articleType == str(row['articleType'])]) >= 800:
                return str(row['articleType'])
        if row['subCategory'] in ['Bottomwear', 'Shoes'] and len(
                df.loc[df.articleType == str(row['articleType'])]) >= 300:
            return str(row['articleType'])
        return str(row['subCategory'])


    df = df.loc[df.masterCategory != 'Sporting Goods']

    exclude_list = ['Vouchers', 'Green', 'Home Furnishing', 'Umbrellas', 'Water Bottle', 'Bath and Body',
                    'Shoe Accessories', 'Sports Accessories', 'Sports Equipment', 'Free Gifts',
                    'Apparel Set']

    for exclude in exclude_list:
        df = df.loc[df.subCategory != exclude]

    tqdm.pandas()
    df['subCategory'] = df.progress_apply(lambda row: category_extractor(row), axis=1)

    df.subCategory = df.subCategory.replace({'Perfumes': 'Green',
                                             'Lips': 'Cosmetic',
                                             'Eyes': 'Cosmetic',
                                             'Wristbands': 'Accessories',
                                             'Cufflinks': 'Accessories',
                                             'Gloves': 'Accessories',
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
except Exception as exp:
    logging.error("Can not train a category classifier", exp)

# season classifier ---------------------------------------------------------
try:
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

except Exception as exp:
    logging.error("Can not train a season classifier", exp)

# gender classifier ---------------------------------------------------------
try:
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

except Exception as exp:
    logging.error("Can not train a gender classifier", exp)
