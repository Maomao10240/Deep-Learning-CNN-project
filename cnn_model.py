import argparse
import numpy as np
def main():

    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', help="The input file", type=str, default='train_val_test_data.pkl')
    parser.add_argument('--ngram', help="Maximum ngram length", type=int, default=3)
    parser.add_argument('--model', help="model name", type=str, default='svc')

    args = parser.parse_args()