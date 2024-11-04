import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import joblib
import math
from binary_classifier import BinaryClassifier, BinaryClassDataset
from multiclass_classifier import ExpandableMultiClassClassifier, MultiClassDataset
from sklearn.neighbors import KNeighborsClassifier


class DecisionModel:
    def __init__(self, multiclass_params, binary_params, multiclass_model_path=None, binary_model_path=None, kNN_model=None, kNN_neighbours=3):
        """
        Initialize the class with a multi-class classifier, binary classifier and a KNN model. Optionally create new models or load pre-trained ones.
        
        Parameters:
            multiclass_params: Dictionary with hyperparameters for multi-class classification model. The dictionary should contain the following keys:
                - "input_size" (int): Size of inputs into the model (required).
                - "layers" (list of int): List of integers specifying size of each layer (required).
                - "num_classes" (int): Number of initial known classes in training dataset (required).
                - "epochs" (int): Number of epochs to train the model for (required only if training from scratch).

            binary_params: Dictionary with hyperparameters for binary classification model. The dictionary should contain the following keys:
                - "input_size" (int): Size of inputs into the model (required).
                - "layers" (list of int): List of integers specifying size of each layer (required).
                - "epochs" (int): Number of epochs to train the model for (required only if training from scratch).

            multiclass_model_path: Path to multi-class classification model (optional if training from scratch).
            
            binary_model_path: Path to binary classification model (optional if training from scratch).
            
            kNN_model: Pre-trained KNN model (optional if training from scratch).
            
            kNN_neighbors: Number of neighbors for KNN (optional if loading model).
        """
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu")

        # Hyperparameters for models
        self.multiclass_params = multiclass_params
        self.binary_params = binary_params

        # Initialize or load multi-class classification model
        self.multiclass_model = ExpandableMultiClassClassifier(
                input_size=self.multiclass_params["input_size"],
                layer_sizes=self.multiclass_params["layers"],
                num_classes=self.multiclass_params["num_classes"])
        
        if multiclass_model_path is not None:
            self.multiclass_model.load_state_dict(torch.load(multiclass_model_path, map_location=self.device))
        
        self.multiclass_model.to(self.device)
        self.multiclass_epochs = self.multiclass_params["epochs"]

        # Initialize or load binary-class classification model
        self.binary_model = BinaryClassifier(
                input_size=self.binary_params["input_size"],
                layer_sizes=self.binary_params["layers"])
        
        if binary_model_path is not None:
            self.binary_model.load_state_dict(torch.load(binary_model_path, map_location=self.device))
        
        self.binary_model.to(self.device)
        self.binary_epochs = self.binary_params["epochs"]

        # Initialize or load KNN model
        if kNN_model is not None:
            self.knn = kNN_model

        else:
            self.knn = KNeighborsClassifier(n_neighbors=kNN_neighbours)

        # Decision thresholds from kNN - IMPORTANT: run self.get_kNN_thresholds before making predictions, make sure labels set correctly to account for simulated novel class
        self.thresholds = None


    def train_multiclass_component(self, train_loader, val_loader, num_epochs, train_num_batches, val_num_batches):
        optimizer = optim.Adam(self.multiclass_model.parameters())
        criterion = nn.CrossEntropyLoss()

        # Tracking history
        train_loss_history = []
        val_loss_history = []

        best_val_loss = float('inf')  # Initialize the best validation loss to infinity
        patience = 5  # Number of epochs to wait before early stopping
        no_improvement_count = 0  # Counter for epochs with no improvement

        overfitting_patience = 10
        prev_val_loss = float('inf')  # To track the previous validation loss
        prev_train_loss = float('inf')  # To track the previous training loss
        overfitting_count = 0

        print("START MULTI-CLASS CLASSIFIER TRAINING")
        for epoch in range(num_epochs):
            print("Epoch [{}/{}]".format(epoch+1, num_epochs))
            # Training phase
            train_loss = 0.0

            for features, labels in train_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.multiclass_model.forward(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() # Sum train losses

            # Average train losses
            avg_train_loss = train_loss / train_num_batches
            train_loss_history.append(avg_train_loss)

            print("TRAIN: Loss: {:.4f}".format(avg_train_loss))

            # Validation phase
            with torch.no_grad():  # Don't compute gradients during validation
                self.multiclass_model.eval()   # Set the classifier to evaluation mode
                val_loss = 0.0

                for features, labels in val_loader:
                    features, labels = features.to(self.device), labels.to(self.device)
                    outputs = self.multiclass_model.forward(features)
                    loss = criterion(outputs, labels)

                    # Sum validation losses
                    val_loss += loss.item()

                # Average validation losses
                avg_val_loss = val_loss / val_num_batches
                val_loss_history.append(avg_val_loss)

                # Print validation results
                print("VAL: Loss: {:.4f}".format(avg_val_loss))

                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    no_improvement_count = 0  # Reset the counter if there is an improvement
                else:
                    no_improvement_count += 1
                    if no_improvement_count >= patience:
                        print("Early stopping triggered. No improvement in validation loss for {} epochs.".format(patience))
                        break

                # Overfitting check
                if avg_val_loss > prev_val_loss and avg_train_loss < prev_train_loss:
                    overfitting_count += 1
                    if overfitting_count >= overfitting_patience:
                        print("Overfitting detected, stopping training")
                        break
                else:
                    overfitting_count = 0
                
                # Update previous losses
                prev_val_loss = avg_val_loss
                prev_train_loss = avg_train_loss

            # Models back to train mode
            self.multiclass_model.train() 


    def train_binary_component(self, train_loader, val_loader, num_epochs, train_num_batches, val_num_batches):
        optimizer = optim.Adam(self.binary_model.parameters())
        criterion = torch.nn.BCEWithLogitsLoss()

        best_val_loss = float('inf')  # Initialize the best validation loss to infinity
        patience = 5  # Number of epochs to wait before early stopping
        no_improvement_count = 0  # Counter for epochs with no improvement

        overfitting_patience = 10
        prev_val_loss = float('inf')  # To track the previous validation loss
        prev_train_loss = float('inf')  # To track the previous training loss
        overfitting_count = 0

        train_loss_history = []
        val_loss_history = []

        print("START BINARY CLASSIFIER TRAINING")
        for epoch in range(num_epochs):
            print("Epoch [{}/{}]".format(epoch+1, num_epochs))
            # Training phase
            train_loss = 0.0

            for features, labels in train_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.binary_model.forward(features)
                loss = criterion(outputs, labels.unsqueeze(1))
                loss.backward()
                optimizer.step()

                # Sum train losses
                train_loss += loss.item()

            # Average train losses
            avg_train_loss = train_loss / train_num_batches
            train_loss_history.append(avg_train_loss)

            print("TRAIN: Loss: {:.4f}".format(avg_train_loss))

            # Validation phase
            with torch.no_grad():  # Don't compute gradients during validation
                self.binary_model.eval()   # Set the classifier to evaluation mode
                val_loss = 0.0

                for features, labels in val_loader:
                    features, labels = features.to(self.device), labels.to(self.device)
                    outputs = self.binary_model.forward(features)
                    loss = criterion(outputs, labels.unsqueeze(1))

                    # Sum validation losses
                    val_loss += loss.item()

                # Average validation losses
                avg_val_loss = val_loss / val_num_batches
                val_loss_history.append(avg_val_loss)

                # Print validation results
                print("VAL: Loss: {:.4f}".format(avg_val_loss))

                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    no_improvement_count = 0  # Reset the counter if there is an improvement
                else:
                    no_improvement_count += 1
                    if no_improvement_count >= patience:
                        print("Early stopping triggered. No improvement in validation loss for {} epochs.".format(patience))
                        break

                # Overfitting check
                if avg_val_loss > prev_val_loss and avg_train_loss < prev_train_loss:
                    overfitting_count += 1
                    if overfitting_count >= overfitting_patience:
                        print("Overfitting detected, stopping training")
                        break
                else:
                    overfitting_count = 0
                
                # Update previous losses
                prev_val_loss = avg_val_loss
                prev_train_loss = avg_train_loss

            # Back to train mode
            self.binary_model.train() 


    def train_kNN_component(self, X_train, y_train):
        print("START KNN TRAINING")
        self.knn.fit(X_train, y_train.values.ravel())


    def get_kNN_thresholds(self, train_data):
        labels = {
            "Benign": 0,
            # "ACK_Flooding": 1,
            "ARP_Spoofing": 1,
            "Port_Scanning": 2,
            "Service_Detection": 3,
            "SYN_Flooding": 4,
            "UDP_Flooding": 5,
            "HTTP_Flooding": 6,
            "Telnet-brute_Force": 7,
            "Host_Discovery": 8,
        } # Edit the label mappings to account for specific attacks being simulated as unknown (ACK Flooding currently set as unknown)
        
        self.thresholds = {}

        for traffic in labels.keys():
            temp = train_data[train_data["label"] == labels[traffic]].reset_index(drop=True)
            X_known_attacks = temp[[col for col in temp.columns if col != "label"]]

            distances, _ = self.knn.kneighbors(X_known_attacks)
            self.thresholds[labels[traffic]] = np.mean(distances)


    def train_all(self, train_data, val_data):
        X_train = train_data[[col for col in train_data.columns if col != "label"]]
        y_train = train_data[["label"]]
        no_novel_val = val_data[val_data["label"] != -1] # Remove novel attacks from validation set for neural network training
        # X_val = val_data[[col for col in val_data.columns if col != "label"]]
        # y_val = val_data[["label"]]

        # Transform DataFrames into data loaders for PyTorch model
        multi_train_dataset = MultiClassDataset(train_data)
        multi_val_dataset = MultiClassDataset(no_novel_val)
        multi_train_loader = DataLoader(multi_train_dataset, batch_size=256, shuffle=True)
        multi_val_loader = DataLoader(multi_val_dataset, batch_size=256, shuffle=True)

        bin_train_dataset = BinaryClassDataset(train_data)
        bin_val_dataset = BinaryClassDataset(no_novel_val)
        bin_train_loader = DataLoader(bin_train_dataset, batch_size=256, shuffle=True)
        bin_val_loader = DataLoader(bin_val_dataset, batch_size=256, shuffle=True)

        train_num_batches = math.ceil(len(multi_train_dataset) / 256)
        val_num_batches = math.ceil(len(multi_val_dataset) / 256)
        
        self.train_multiclass_component(multi_train_loader, multi_val_loader, self.multiclass_epochs, train_num_batches, val_num_batches)
        self.train_binary_component(bin_train_loader, bin_val_loader, self.binary_epochs, train_num_batches, val_num_batches)
        self.train_kNN_component(X_train, y_train)


    def save_models(self, multiclass_path="multiclass_classifier.pt", binary_path="binary_classifier.pt", kNN_path="kNN.joblib"):
        """
        Saves each model to the specified paths.

        Parameters:
            multiclass_path: Path to save multiclass classifier.
            binary_path: Path to save binary classifier.
            kNN_path: Path to save kNN classifier.
        """
        torch.save(self.multiclass_model.state_dict(), multiclass_path)
        torch.save(self.multiclass_model.state_dict(), binary_path)
        joblib.dump(self.knn, kNN_path)


    def multiclass_predict(self, test_loader):
        y_pred_class = []
        y_pred_prob = []

        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(self.device), labels.to(self.device)

                # Forward passes
                output = self.multiclass_model.forward(features)

                # Get predicted labels for classification
                _, predicted = torch.max(output, 1)
                probabilities = torch.softmax(output, dim=1)

                # Collect predicted labels and probabilities
                y_pred_class.extend(predicted.cpu().numpy())
                y_pred_prob.extend(probabilities.gather(1, predicted.view(-1, 1)).cpu().numpy())  # Gather probabilities for the predicted classes

        return y_pred_class, y_pred_prob


    def binary_predict(self, test_loader):
        y_pred_class = []
        y_pred_prob = []

        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(self.device), labels.to(self.device)

                # Forward passes
                output = self.binary_model.forward(features)

                # Apply sigmoid to convert logits to probabilities
                probabilities = torch.sigmoid(output)
                
                # Convert probabilities to binary predictions (0 or 1)
                predicted = (probabilities >= 0.5).long()

                # Collect true and predicted labels
                y_pred_class.extend(predicted.cpu().numpy())
                y_pred_prob.extend(probabilities.cpu().numpy())

        return y_pred_class, y_pred_prob


    def kNN_predict(self, X_test):
        return self.knn.predict(X_test), np.max(self.knn.predict_proba(X_test), axis=1)
    

    def kNN_mean_neighbour_distances(self, X_test):
        distances, _ = self.knn.kneighbors(X_test)
        return distances.mean(axis=1)


    def full_predict(self, train_data, test_data):
        X_test = test_data[[col for col in test_data.columns if col != "label"]]
        y_test = test_data[["label"]]

        test_batch_size = 256
        multiclass_test_dataset = MultiClassDataset(test_data)
        multiclass_test_loader = DataLoader(multiclass_test_dataset, batch_size=test_batch_size, shuffle=False)

        binary_test_dataset = BinaryClassDataset(test_data)
        binary_test_loader = DataLoader(binary_test_dataset, batch_size=test_batch_size, shuffle=False)

        multiclass_pred_class, multiclass_pred_prob = self.multiclass_predict(multiclass_test_loader)
        binary_pred_class, binary_pred_prob = self.binary_predict(binary_test_loader)
        knn_pred_class, knn_pred_prob = self.kNN_predict(X_test)
        knn_mean_dist = self.kNN_mean_neighbour_distances(X_test)

        self.get_kNN_thresholds(train_data)
        
        # Using sub-model predictions, label data as -1 if novel, else give class label
        final_labels = []
        for i in range(len(X_test)):
            multi_label = multiclass_pred_class[i]
            binary_label = binary_pred_class[i]
            knn_label = knn_pred_class[i]

            if (
                (multi_label == 0 and binary_label == 0 and knn_label == 0) # All models predict and agree on benign
                or (multi_label == 0 and binary_label == 0 and knn_label > 0) # Most models predict benign, probably benign
                or (multi_label > 0 and binary_label == 0 and knn_label == 0) # Binary and knn agree
            ): 
                final_labels.append(0) # Predict benign
            else:
                # Determine what attack class to predict
                if knn_mean_dist[i] > self.thresholds.get(knn_label) * 2.5:
                    final_labels.append(-1) # Novel attack
                else:
                    final_labels.append(knn_label) # Known attack

        return final_labels


    # def update(self):
    #     #TODO
