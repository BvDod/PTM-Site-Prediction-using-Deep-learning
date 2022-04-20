import torch
import torch.nn as nn

class MyIter(object):
    """An iterator."""
    def __init__(self, my_loader):
        self.my_loader = my_loader
        self.loader_iters = [iter(loader) for loader in self.my_loader.loaders]

    def __iter__(self):
        return self

    def __next__(self):
        # When the shortest loader (the one with minimum number of batches)
        # terminates, this iterator will terminates.
        # The `StopIteration` raised inside that shortest loader's `__next__`
        # method will in turn gets out of this `__next__` method.
        batches = [loader_iter.next() for loader_iter in self.loader_iters]
        return self.my_loader.combine_batch(batches)

    def __len__(self):
        return len(self.my_loader)


class MultiTaskLoader(object):
    """This class wraps several pytorch DataLoader objects, allowing each time 
    taking a batch from each of them and then combining these several batches 
    into one. This class mimics the `for batch in loader:` interface of 
    pytorch `DataLoader`.
    Args: 
        loaders: a list or tuple of pytorch DataLoader objects
    """
    def __init__(self, loaders):
        self.loaders = loaders

    def __iter__(self):
        return MyIter(self)

    def __len__(self):
        return min([len(loader) for loader in self.loaders])

    # Customize the behavior of combining batches here.
    def combine_batch(self, batches):
        task_indexes = [sum(x[0].shape[0] for x in batches[:i+1]) for i in range(len(batches))]
        task_indexes = [0,] + task_indexes

        features = torch.cat([batch[0] for batch in batches], dim=0)
        labels = torch.cat([batch[1] for batch in batches])

        return [features, labels, task_indexes]


class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num):
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))

    def forward(self, list_of_losses):
        
        total_loss = 0
        for i, loss in enumerate(list_of_losses):
            precision = torch.exp(-self.log_vars[i])
            loss = precision*loss + self.log_vars[i]
            total_loss += loss

        
        return total_loss