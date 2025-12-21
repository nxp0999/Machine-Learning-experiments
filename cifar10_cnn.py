"""
CS 6375 Project 3 
CNN Implementation on CIFAR-10 (PART 2)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
import torchvision
import torchvision.transforms as transforms
import time
import json

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_cifar10_data():
    transform = transforms.Compose([transforms.ToTensor()])

    cifar_train_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    cifar_train, cifar_val = random_split(cifar_train_full, [45000, 5000], generator=torch.Generator().manual_seed(42))

    datasets = {'cifar10': {'train': cifar_train, 'val': cifar_val, 'test': cifar_test, 'channels': 3, 'num_classes': 10}}
    print(f"CIFAR-10 - Train: {len(cifar_train)}, Val: {len(cifar_val)}, Test: {len(cifar_test)}")
    return datasets

class BaselineCNN(nn.Module):
    def __init__(self, num_channels=3, num_classes=10, dropout=0.2):
        super(BaselineCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.flatten = nn.Flatten()
        self.fc_layers = None
        self.dropout = dropout
        self.num_classes = num_classes

    def _initialize_fc(self, x):
        if self.fc_layers is None:
            x = self.conv_layers(x)
            x = self.flatten(x)
            self.fc_layers = nn.Sequential(
                nn.Linear(x.shape[1], 128), nn.ReLU(), nn.Dropout(self.dropout), nn.Linear(128, self.num_classes)
            ).to(x.device)

    def forward(self, x):
        if self.fc_layers is None:
            self._initialize_fc(x)
        return self.fc_layers(self.flatten(self.conv_layers(x)))

class EnhancedCNN(nn.Module):
    def __init__(self, num_channels=3, num_classes=10, dropout=0.2):
        super(EnhancedCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout2d(0.1)
        )
        self.flatten = nn.Flatten()
        self.fc_layers = None
        self.dropout = dropout
        self.num_classes = num_classes

    def _initialize_fc(self, x):
        if self.fc_layers is None:
            x = self.conv_layers(x)
            x = self.flatten(x)
            self.fc_layers = nn.Sequential(
                nn.Linear(x.shape[1], 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(self.dropout), nn.Linear(128, self.num_classes)
            ).to(x.device)

    def forward(self, x):
        if self.fc_layers is None:
            self._initialize_fc(x)
        return self.fc_layers(self.flatten(self.conv_layers(x)))

class DeeperCNN(nn.Module):
    def __init__(self, num_channels=3, num_classes=10, dropout=0.2):
        super(DeeperCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout2d(0.2)
        )
        self.flatten = nn.Flatten()
        self.fc_layers = None
        self.dropout = dropout
        self.num_classes = num_classes

    def _initialize_fc(self, x):
        if self.fc_layers is None:
            x = self.conv_layers(x)
            x = self.flatten(x)
            self.fc_layers = nn.Sequential(
                nn.Linear(x.shape[1], 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(self.dropout),
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(self.dropout),
                nn.Linear(128, self.num_classes)
            ).to(x.device)

    def forward(self, x):
        if self.fc_layers is None:
            self._initialize_fc(x)
        return self.fc_layers(self.flatten(self.conv_layers(x)))

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / len(train_loader), 100. * correct / total

def evaluate(model, data_loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / len(data_loader), 100. * correct / total

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, patience=3, device='cuda'):
    model = model.to(device)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc, best_model_state, epochs_no_improve = 0.0, None, 0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Train: {train_acc:.2f}%, Val: {val_acc:.2f}%")

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if best_model_state:
        model.load_state_dict(best_model_state)
    return history, best_val_acc

def get_hyperparams():
    return [
        {'lr': 0.001, 'batch_size': 128, 'optimizer': 'Adam', 'dropout': 0.2},
        {'lr': 0.01, 'batch_size': 128, 'optimizer': 'Adam', 'dropout': 0.2},
        {'lr': 0.0001, 'batch_size': 128, 'optimizer': 'Adam', 'dropout': 0.2},
        {'lr': 0.001, 'batch_size': 64, 'optimizer': 'Adam', 'dropout': 0.2},
        {'lr': 0.001, 'batch_size': 128, 'optimizer': 'SGD', 'dropout': 0.2},
        {'lr': 0.001, 'batch_size': 128, 'optimizer': 'Adam', 'dropout': 0.5},
        {'lr': 0.001, 'batch_size': 128, 'optimizer': 'Adam', 'dropout': 0.0},
        {'lr': 0.01, 'batch_size': 64, 'optimizer': 'SGD', 'dropout': 0.3},
    ]

def run_hyperparameter_search(model_class, model_name, dataset_name, datasets, hyperparams_list, num_epochs=10, device='cuda'):
    print(f"\n{'='*70}\n{model_name} on {dataset_name.upper()}\n{'='*70}")
    dataset_info = datasets[dataset_name]
    results = []

    for i, config in enumerate(hyperparams_list):
        print(f"\nConfig {i+1}/{len(hyperparams_list)} - LR: {config['lr']}, Batch: {config['batch_size']}")

        train_loader = DataLoader(dataset_info['train'], batch_size=config['batch_size'], shuffle=True, num_workers=2)
        val_loader = DataLoader(dataset_info['val'], batch_size=config['batch_size'], shuffle=False, num_workers=2)

        model = model_class(num_channels=dataset_info['channels'], num_classes=dataset_info['num_classes'], dropout=config['dropout'])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9) if config['optimizer'] == 'SGD' else optim.Adam(model.parameters(), lr=config['lr'])

        start_time = time.time()
        history, best_val_acc = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, 3, device)
        runtime = (time.time() - start_time) / 60

        results.append({'config': config, 'best_val_acc': best_val_acc, 'runtime': runtime, 'history': history, 'model_state': model.state_dict().copy()})
        print(f"Best Val Acc: {best_val_acc:.2f}% - Runtime: {runtime:.2f} min")

    return results

def find_best_config(results):
    best_idx = max(range(len(results)), key=lambda i: results[i]['best_val_acc'])
    return best_idx, results[best_idx]

def print_results_table(all_results, dataset_name):
    print(f"\n{'='*100}\nResults Summary - {dataset_name.upper()}\n{'='*100}")
    for arch_name, results in all_results.items():
        print(f"\n{arch_name}:")
        print("-" * 100)
        for i, result in enumerate(results):
            config = result['config']
            print(f"{i+1:<8} {config['lr']:<10} {config['batch_size']:<8} {config['optimizer']:<10} {config['dropout']:<10} {result['best_val_acc']:<12.2f} {result['runtime']:<15.2f}")
        best_idx, best_result = find_best_config(results)
        print(f"BEST: #{best_idx+1} - {best_result['best_val_acc']:.2f}%\n")

def save_results(all_results, dataset_name):
    serializable = {}
    for arch_name, results in all_results.items():
        serializable[arch_name] = [{'config': r['config'], 'best_val_acc': float(r['best_val_acc']),
                                    'runtime': float(r['runtime'])} for r in results]

    filename = f'{dataset_name}_cnn_results.json'
    with open(filename, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"Results saved to {filename}")

def compute_final_test_accuracy(model_class, best_config, datasets, dataset_name, device='cuda'):
    print(f"\n{'='*70}\nFinal Test Evaluation\n{'='*70}")
    dataset_info = datasets[dataset_name]
    combined_train = ConcatDataset([dataset_info['train'], dataset_info['val']])

    train_loader = DataLoader(combined_train, batch_size=best_config['batch_size'], shuffle=True, num_workers=2)
    test_loader = DataLoader(dataset_info['test'], batch_size=best_config['batch_size'], shuffle=False, num_workers=2)

    model = model_class(num_channels=dataset_info['channels'], num_classes=dataset_info['num_classes'], dropout=best_config['dropout']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=best_config['lr'], momentum=0.9) if best_config['optimizer'] == 'SGD' else optim.Adam(model.parameters(), lr=best_config['lr'])

    print("Training on train+val...")
    for epoch in range(10):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}/10 - Acc: {train_acc:.2f}%")

    _, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    return test_acc

def main():
    print("\n" + "#"*70)
    print("# CS 6375 Project 3 - PART 2: CIFAR-10 CNN EXPERIMENTS")
    print("#"*70)

    datasets = load_cifar10_data()
    hyperparams = get_hyperparams()

    start_total = time.time()

    cifar_results = {}
    cifar_results['CNN (baseline)'] = run_hyperparameter_search(BaselineCNN, 'Baseline CNN', 'cifar10', datasets, hyperparams, 10, device)
    cifar_results['CNN (enhanced)'] = run_hyperparameter_search(EnhancedCNN, 'Enhanced CNN', 'cifar10', datasets, hyperparams, 10, device)
    cifar_results['CNN (deeper)'] = run_hyperparameter_search(DeeperCNN, 'Deeper CNN', 'cifar10', datasets, hyperparams, 10, device)

    print_results_table(cifar_results, 'cifar10')
    save_results(cifar_results, 'cifar10')

    best_arch = max(cifar_results.items(), key=lambda x: find_best_config(x[1])[1]['best_val_acc'])
    _, best_result = find_best_config(best_arch[1])

    model_class = BaselineCNN if 'baseline' in best_arch[0] else (EnhancedCNN if 'enhanced' in best_arch[0] else DeeperCNN)
    cifar_test_acc = compute_final_test_accuracy(model_class, best_result['config'], datasets, 'cifar10', device)

    total_time = (time.time() - start_total) / 60

    print("\n" + "="*70)
    print("CIFAR-10 CNNs COMPLETE")
    print("="*70)
    print(f"Final Test Accuracy: {cifar_test_acc:.2f}%")
    print(f"Total Runtime: {total_time:.2f} minutes")
    print("="*70)

if __name__ == "__main__":
    main()