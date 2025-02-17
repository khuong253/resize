from .publaynet import PubLayNetDataset
from .rico import Rico25Dataset
from .obello import ObelloDataset

_DATASETS = [
    Rico25Dataset,
    PubLayNetDataset,
    ObelloDataset,
]
DATASETS = {d.name: d for d in _DATASETS}
