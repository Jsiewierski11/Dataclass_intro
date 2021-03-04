from dataclasses import dataclass, asdict
import json, yaml
import argparse

"""
Implementation of the Vocabulary class (Example 3-15) from NLP with Pytorch as a dataclass
"""
 # TODO: Figure out how to implement serializable funtions, is it needed?

@dataclass
class Vocabulary:
    token_to_idx: dict = None
    add_unk: bool = True
    unk_token: str = "<UNK>"
    
    def __post_init__(self):
        """
        Args:
            token_to_idx (dict): a pre-existing map of topkens to indicies.
            add_unk (bool): a flag that indicates whether to add the UNK token.
            unk_token (str): the UNK token to add into the Vocabulary.
        """
        if self.token_to_idx is None:
            self.token_to_idx = {}

        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}

        self.unk_index = -1
        if self.add_unk:
            self.unk_index = self.add_token(self.unk_token)

    def add_token(self, token):
        """
        Update maping dicts based on the token.

        Args:
            token (str): the item to add into the Vocabulary.
        Returns:
            index (int): the integer corresponging to the token. 
        """

        if token in self.token_to_idx:
            index = self.token_to_idx
        else:
            index = len(self.token_to_idx)
            self.token_to_idx[token] = index
            self.idx_to_token[index] = token
        return index

    def lookup_token(self, token):
        """self.token_to_idx
        Retrieve the index associated with the token or the UNK index if token isn't present.
        
        Args:self.token_to_idx
            token (str): the token to look up.
        Returns: 
            index (int): the index corresponding to the token.
        Note:self.token_to_idx
            'unk_index' needs to be >=0 (having been added into the vocabulary for the UNK functionality)
        """

        if self.add_unk:
            return self.token_to_idx.get(token, self.unk_index)
        else:
            return self.token_to_idx[token]

    def lookup_index(self, index):
        """
        Retrieve the token associated with the token or the UNK index if token isn't present.
        
        Args:
            token (str): the index to look up.
        Returns: 
            index (int): the token corresponding to the index.
        Raises:
            Keyerror: if the index is not in the Vocabulary.
        """

        if index not in self.idx_to_token:
            raise Keyerror("the index (%d) is not in the Vocabulary" % index)
        return self.idx_to_token[index]

    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)
    
    def __len__(self):
        return len(self.token_to_idx)

   



if __name__ == "__main__":
    tokens = {"How": 1, "are": 2, "you": 3}
    vocab = Vocabulary(tokens, True, "<UNK>")

    print("Index for token 'how': {}".format(vocab.lookup_token("How")))
    print("Index for token 'you': {}".format(vocab.lookup_token("you")))
    print("Token at index '2': {}".format(vocab.lookup_index(2)))