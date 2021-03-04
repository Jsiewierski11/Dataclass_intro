import json, yaml
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
import torch.optim as optim

from review.review_classifier import ReviewClassifier
from review.review_dataset import ReviewDataset

def make_train_state():
    return {'epoch_index': 0, 
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': -1,
            'test_acc': -1}

# NOTE: Named namespace to be consistent with what is in the book.
@dataclass
class namespace:
    frequency: int
    model_state_file: str
    review_csv: str
    save_dir: str
    vectorizer_file: str
    batch_size: int
    learning_rate: int
    num_epochs: int
    seed: int

    @classmethod
    def from_json(cls, json_file):
        config_json = json.load(open(json_file))
        return cls(**config_json)

    @classmethod
    def from_yaml(cls, yaml_file):
        config_yaml = yaml.load(open(yaml_file), yaml.FullLoader)
        return cls(**config_yaml)




if __name__ == "__main__":
    # simple_config = TextGenerationConfig.from_json("../config_files/simple_generation_config.json")
    args = namespace.from_yaml("./config/program_options.yml")
    train_state = make_train_state()
    print("Train state: {}".format(train_state))

    # Check for GPU
    if not torch.cuda.is_available():
        args.cuda = False
    args.device = torch.device("cuda" if args.cuda else "cpu")

    # Dataset and vectorizer
    dataset = ReviewDataset.load_dataset_and_make_vectorizer(args.review_csv)
    print("Dataset: {}".format(dataset))
    vectorizer = dataset.get_vectorizer()
    print("Created dataset and vectorizer...")

    # Model
    classifier = ReviewClassifier(num_features=len(vectorizer.review_vocab))
    classifier = classifier.to(args.device)
    print("Created classifier...")
    
    # Loss and optimizer
    loss_func = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)
    print("Created loss_func and optimizer")

    # TODO: Write training loop.
    # for epoch_index in range(args.num_epochs)
        