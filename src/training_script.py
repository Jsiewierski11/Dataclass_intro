import json, yaml
imort torch
imort torch.nn as nn
import torch.optim as optim

from review.review_classifier import ReviewClassifier
from review.review_dataset import ReviewDataset

def make_train_state(args):
    return {'epoch_index': 0, 
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': -1,
            'test_acc': -1}



if __name__ == "__main__":
    # simple_config = TextGenerationConfig.from_json("../config_files/simple_generation_config.json")
    args = TextGenerationConfig.from_yaml("./config/program_options.yml")
    print("Train state: {}".format(args))

    # Check for GPU
    if not torch.cuda.is_available():
        args.cuda = False
    args.device = torch.device("cuda" if args.cuda else "cpu")

    # Dataset and vectorizer
    dataset = ReviewDataset.load_dataset_and_make_vectorizer(args.review_csv)
    vectorizer = dataset.get_vectorizer()

    # Model
    classifier = ReviewClassifier(num_features=len(vectorizer.review_vocab))
    classifier = classifier.to(args.device)
    
    # Loss and optimizer
    loss_func = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)

    # TODO: Write training loop.
    # for epoch_index in range(args.num_epochs)
        