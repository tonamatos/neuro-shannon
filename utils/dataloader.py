from torch_geometric.loader import DataLoader

def dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )