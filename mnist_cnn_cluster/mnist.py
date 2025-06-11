import torch
from torch import nn, optim
from torchsummary import summary
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_mnist_data(device, csv_file='mnist_data.csv', test_size=0.2, val_size=0.1, random_state=42):
    """
    Load MNIST data from local CSV file and split into train/validation/test sets.
    
    Args:
        device: PyTorch device (CPU/GPU)
        csv_file: Path to CSV file containing MNIST data
        test_size: Proportion of data for test set
        val_size: Proportion of training data for validation set
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test) as PyTorch tensors
    """
    print(f"Loading MNIST data from {csv_file}...")
    
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Separate features (pixels) from labels (class)
    pixel_columns = [col for col in df.columns if col.startswith('pixel')]
    X = df[pixel_columns].values.astype('float32')
    y = df['class'].values.astype(int)
    
    print(f"Data loaded: {X.shape[0]} samples with {X.shape[1]} features")

    # Reshape to 28x28 images and normalize to [0, 1]
    X = X.reshape(-1, 28, 28, 1) / 255.0

    # Split into train+val and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Split train into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, stratify=y_train, random_state=random_state
    )

    # Convert labels to PyTorch tensors
    y_train = torch.tensor(y_train, device=device, dtype=torch.long)
    y_val = torch.tensor(y_val, device=device, dtype=torch.long)
    y_test = torch.tensor(y_test, device=device, dtype=torch.long)

    # Convert images to PyTorch tensors with proper channel ordering
    # Change from (batch, height, width, channels) to (batch, channels, height, width)
    X_train = torch.tensor(X_train, device=device, dtype=torch.float32).permute(0, 3, 1, 2)
    X_val = torch.tensor(X_val, device=device, dtype=torch.float32).permute(0, 3, 1, 2)
    X_test = torch.tensor(X_test, device=device, dtype=torch.float32).permute(0, 3, 1, 2)
    
    print(f"Training: {X_train.size(0)}, Validation: {X_val.size(0)}, Test: {X_test.size(0)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_mnist_cnn(num_classes=10):
    """
    Functional version of the MNIST CNN model using nn.Sequential.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(7 * 7 * 64, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, num_classes),
        nn.Softmax(dim=1)
    )


def calculate_accuracy(predictions, targets):
    """Calculate accuracy from predictions and targets."""
    batch_size = targets.size(0)
    correct = (predictions == targets).sum().item()
    return correct / batch_size


def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32, learning_rate=0.001):
    """
    Train the CNN model.
    
    Args:
        model: PyTorch model to train
        X_train, y_train: Training data and labels
        X_val, y_val: Validation data and labels
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
    
    Returns:
        Dictionary containing training history
    """
    num_batches = (X_train.size(0) + batch_size - 1) // batch_size
    
    print(f"Training model with {X_train.size(0)} samples for {epochs} epochs")
    print(f"Each epoch has {num_batches} batches of size {batch_size}")
    print(f"Validating with {X_val.size(0)} samples")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training history
    history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}
    
    # Training loop
    for epoch in tqdm(range(epochs), desc='Training Progress'):
        model.train()  # Set model to training mode
        epoch_loss = 0.0
        epoch_acc = 0.0
        
        # Mini-batch training
        for batch_start in range(0, X_train.size(0), batch_size):
            batch_end = min(batch_start + batch_size, X_train.size(0))
            X_batch = X_train[batch_start:batch_end]
            y_batch = y_train[batch_start:batch_end]

            # Forward pass
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            predictions = torch.argmax(outputs, dim=1)
            accuracy = calculate_accuracy(predictions, y_batch)
            
            epoch_loss += loss.item()
            epoch_acc += accuracy

        # Calculate average metrics for epoch
        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches
        
        # Validation
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
            val_predictions = torch.argmax(val_outputs, dim=1)
            val_acc = calculate_accuracy(val_predictions, y_val)
        
        # Store history
        history['loss'].append(avg_loss)
        history['acc'].append(avg_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return history


def evaluate_model(model, X_test, y_test, batch_size=256):
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained PyTorch model
        X_test, y_test: Test data and labels
        batch_size: Batch size for evaluation
    """
    model.eval()
    all_predictions = []
    
    print(f"Evaluating model on {X_test.size(0)} test samples")
    
    with torch.no_grad():
        for batch_start in tqdm(range(0, X_test.size(0), batch_size), desc="Evaluating"):
            batch_end = min(batch_start + batch_size, X_test.size(0))
            X_batch = X_test[batch_start:batch_end]
            
            outputs = model(X_batch)
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.append(predictions.cpu().numpy())
    
    # Combine all predictions
    all_predictions = np.concatenate(all_predictions)
    y_test_np = y_test.cpu().numpy()
    
    # Calculate and print metrics
    test_accuracy = accuracy_score(y_test_np, all_predictions)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test_np, all_predictions)}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test_np, all_predictions)}")


def plot_training_history(history):
    """Plot training and validation loss and accuracy over time."""
    epochs = len(history['loss'])
    print(f"Plotting training history for {epochs} epochs")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot loss
    ax1.plot(history['loss'], label='Training Loss', color='blue')
    ax1.plot(history['val_loss'], label='Validation Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history['acc'], label='Training Accuracy', color='blue')
    ax2.plot(history['val_acc'], label='Validation Accuracy', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("Training history plot saved as 'training_history.png'")
    plt.show()


def main():
    """Main function to run the complete MNIST training pipeline."""
    # Set device and random seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    print("Random seed set to 42")
    
    # 1. Load data
    print("\n" + "="*50)
    print("LOADING DATA")
    print("="*50)
    X_train, X_val, X_test, y_train, y_val, y_test = load_mnist_data(device)
    
    # 2. Create model
    print("\n" + "="*50)
    print("INITIALIZING MODEL")
    print("="*50)
    model = create_mnist_cnn().to(device)
    print("CNN Model initialized")
    
    # Print model summary
    try:
        summary(model, input_size=(1, 28, 28))
    except:
        print("Model summary not available (torchsummary may not be installed)")
    
    # 3. Train model
    print("\n" + "="*50)
    print("TRAINING MODEL")
    print("="*50)
    history = train_model(
        model, X_train, y_train, X_val, y_val, 
        epochs=35, batch_size=64, learning_rate=0.001
    )
    
    # 4. Evaluate model
    print("\n" + "="*50)
    print("EVALUATING MODEL")
    print("="*50)
    evaluate_model(model, X_test, y_test)
    
    # 5. Plot training history
    print("\n" + "="*50)
    print("PLOTTING RESULTS")
    print("="*50)
    plot_training_history(history)
    
    print("\n" + "="*50)
    print("ALL DONE!")
    print("="*50)


if __name__ == "__main__":
    main()