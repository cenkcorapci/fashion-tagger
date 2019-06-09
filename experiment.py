import argparse
from experiments.fashion_tagger_experiment import FashionTaggerExperiment

usage_docs = """
--epochs <integer> Number of epochs
--val_split <float> Set validation split(between 0 and 1)
"""

parser = argparse.ArgumentParser(usage=usage_docs)

parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--val_split', type=float, default=.1)

args = parser.parse_args()

experiment = FashionTaggerExperiment(nb_epochs=args.epochs, val_split=args.val_split)
experiment.train_model()
