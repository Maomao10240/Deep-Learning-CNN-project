import pickle
import argparse

#


def main():

    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', help="The input file", type=str, default='data_no_batch.h5')
    parser.add_argument('--ngram', help="Maximum ngram length", type=int, default=3)
    args = parser.parse_args()
 
    #Load data from pickle file
    file_path = 'data.pickle'
    train_text, train_target, val_text, val_target, test_text, test_target = pd.read_pickle(args.data)

    with open("./df_annot_text_split.pkl", 'rb') as f:
        data = pickle.load(f)

    # Display the loaded data
    for line in data[:5]:
        print(line)


    # all_accs = []
    # all_aucs = []

    # with h5py.File(args.data, "r") as f:
    #     if not os.path.isfile(os.path.join("converted/X_features-"+str(args.ngram)+".npz")):
    #         print('Computing features.')
    #         train_x = f["train"][:]
    #         valid_x = f["val"][:]
    #         test_x = f["test"][:]

    #         X_features, val_X_features, test_X_features = extract_features(train_x, valid_x, test_x, args.ngram)
    #     else:
    #         print('Skipping Build of Ngrams: model already exists.')
    #         X_features = dict(np.load(os.path.join("converted", "X_features-"+str(args.ngram)+".npz")))['features'].item()
    #         val_X_features = dict(np.load(os.path.join("converted", "val_X_features-"+str(args.ngram)+".npz")))['features'].item()
    #         test_X_features = dict(np.load(os.path.join("converted", "test_X_features-"+str(args.ngram)+".npz")))['features'].item()




    #     for index, condition in enumerate(conditions):
    #         # for dataset_filepath in sorted(glob.glob(os.path.join(data_folder_formatted, 'icu_frequent_flyers_cohort.npz'))):

    #         print('Current Condition: {0}'.format(condition))

    #         train_y = f["train_label"][:,index]
    #         valid_y = f["val_label"][:,index]
    #         test_y = f["test_label"][:,index]
    #         current_accs = []
    #         current_aucs = []
    #         #for subset in xrange(1,21):
    #         #    s = subset/float(20)
    #         acc, auc = make_predictions(X_features, train_y, val_X_features, valid_y, test_X_features, test_y, 1)#s)
    #         current_accs.append(acc)
    #         current_aucs.append(auc)
    #         print('\n')
          

if __name__ == "__main__":
    main()
