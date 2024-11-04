import torch


class ExpandableMultiClassClassifier(torch.nn.Module):
    def __init__(self, input_size, layer_sizes, num_classes):
        super(ExpandableMultiClassClassifier, self).__init__()
        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.num_classes = num_classes
        self.model = self._create_classifier()

    def _create_classifier(self):
        layers = []
        in_size = self.input_size

        for out_size in self.layer_sizes:
            layers.append(torch.nn.Linear(in_size, out_size))
            layers.append(torch.nn.ReLU())
            in_size = out_size
        
        # Final classification layer
        layers.append(torch.nn.Linear(in_size, self.num_classes))
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
    def expand_classes(self, num_new_classes):
        self.num_classes = num_new_classes

        # Expand the final layer to accommodate new classes
        current_layers = list(self.model.children())
        last_layer = current_layers[-1]
        
        # Preserve the old weights and biases
        old_weight = last_layer.weight.data
        old_bias = last_layer.bias.data
        
        # Create new weights and biases for the new classes
        new_weight = torch.cat([old_weight, torch.zeros((num_new_classes, old_weight.size(1)))], dim=0)
        new_bias = torch.cat([old_bias, torch.zeros(num_new_classes)], dim=0)
        
        # Update the final layer with the expanded weights and biases
        new_layer = torch.nn.Linear(old_weight.size(1), old_weight.size(0) + num_new_classes)
        new_layer.weight.data = new_weight
        new_layer.bias.data = new_bias
        
        # Replace the old final layer with the new one
        current_layers[-1] = new_layer
        self.model = torch.nn.Sequential(*current_layers)
    

class MultiClassDataset(torch.utils.data.Dataset):
    def __init__(self, file):
        # Load in datasets as Pandas dataframe (from csv)
        self.data = file

        # Feature columns
        self.dataset_features = [col for col in self.data.columns if (col != "label")]

        self.input_size = len(self.dataset_features)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        features_tensor = torch.tensor(sample[self.dataset_features].values, dtype=torch.float32)
        labels_tensor = torch.tensor(sample['label']).long()

        return features_tensor, labels_tensor