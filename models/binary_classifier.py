import torch


class BinaryClassifier(torch.nn.Module):
    def __init__(self, input_size, layer_sizes):
        super(BinaryClassifier, self).__init__()
        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.model = self._create_classifier()

    def _create_classifier(self):
        layers = []
        in_size = self.input_size

        for out_size in self.layer_sizes:
            layers.append(torch.nn.Linear(in_size, out_size))
            layers.append(torch.nn.ReLU())
            in_size = out_size
        
        # Final binary classification layer (no Sigmoid, using BCEWithLogitsLoss criterion)
        layers.append(torch.nn.Linear(in_size, 1))

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    

class BinaryClassDataset(torch.utils.data.Dataset):
    def __init__(self, file):
        # Load in datasets
        self.data = file

        # Feature columns
        self.dataset_features = [col for col in self.data.columns if (col != "label")]

        self.input_size = len(self.dataset_features)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        features_tensor = torch.tensor(sample[self.dataset_features].values, dtype=torch.float32)

        binary_label = 0 if sample['label'] == 0 else 1
        label_tensor = torch.tensor(binary_label).float()

        return features_tensor, label_tensor