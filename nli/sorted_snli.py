import torchtext
from torchtext.datasets import SNLI

class SortedSNLI(SNLI):

    @staticmethod
    def sort_key(ex):
        return torchtext.data.interleave_keys(
            len(ex.premise), len(ex.hypothesis))

    @classmethod
    def iters(cls, batch_size=32, device=0, root='.data',
              vectors=None, trees=False, **kwargs):
        if trees:
            TEXT = torchtext.datasets.nli.ParsedTextField()
            TRANSITIONS = torchtext.datasets.nli.ShiftReduceField()
        else:
            TEXT = torchtext.data.Field(tokenize='spacy')
            TRANSITIONS = None
        LABEL = torchtext.data.Field(sequential=False)

        train, val, test = cls.splits(
            TEXT, LABEL, TRANSITIONS, root=root, **kwargs)

        TEXT.build_vocab(train, vectors=vectors)
        LABEL.build_vocab(train)

        return torchtext.data.BucketIterator.splits(
            (train, val, test), batch_size=batch_size, device=device, sort=True)
