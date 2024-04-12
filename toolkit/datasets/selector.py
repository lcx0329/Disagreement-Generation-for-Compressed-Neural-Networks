import numpy as np
from torch.utils.data import DataLoader, Dataset
from . import wrapper

class Selector:
    
    def select(self, datasource, k):
        raise NotImplementedError


class RandomSelector(Selector):
    
    def select(self, datasource, k):
        return np.random.permutation(len(datasource))[: k]


class UncertaintyBasedSelector(Selector):
    """
        indices = type(
            'EntropyBasedSelector', 
            (UncertaintyBasedSelector, ),  
            {"get_uncertainty": lambda self, probs: Uncertainty.entropy(probs)}
        )(model_ori).select(dataset, k, 4)
      
    """
    def __init__(self) -> None:
        super().__init__()
    
    def select(self, datasource, k, T=1):
        if isinstance(datasource, Dataset):
            datasource: DataLoader = wrapper.dataset_to_loader(datasource)
            
        values = self.get_uncertainty(datasource)
        
        indices = np.argsort(values)
        return indices[: k]
        
    def get_uncertainty(self, datasource):
        raise NotImplementedError
    
    def get_class(func):
        return type(
            'UnKnown',
            (UncertaintyBasedSelector, ),
            {"get_uncertainty": lambda self, datasource: func(datasource)}
        )
