import numpy as np
import torch

def create_sequences(data, seq_lenght):
    X, y = [], []
    for i in range(len(data) - seq_lenght):
        X.append(data[i:(i+seq_lenght)])
        y.append(data[i+seq_lenght])
    return np.array(X), np.array(y)

def train_test_split(X, y, test_size=0.33):
    train_size = int(len(y) * (1 - test_size))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size],y[train_size:]

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()

    return X_train, X_test, y_train, y_test


def create_loader(X, y, batch_size=64, shuffle=True):
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle)
    return loader

def one_epoch_train(model, loader, device, loss_fn, optimizer):
    total_loss = 0.0
    model.train()
    for batch_X, batch_y in loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        predictions = model(batch_X)
        loss = loss_fn(predictions, batch_y)
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
        total_loss += loss.item()
 
    average_loss = total_loss / len(loader)
    return average_loss

def one_epoch_val(model, loader, device, loss_fn):
    model.eval()
    
    with torch.no_grad():
        total_loss = 0.0
 
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            predictions_test = model(batch_X)
            test_loss = loss_fn(predictions_test, batch_y)
 
            total_loss += test_loss.item()
 

        average_loss = total_loss / len(loader)
        return average_loss
    
def train(num_epochs, model, train_loader, test_loader, device, loss_fn, optimizer):
    train_hist =[]
    test_hist =[]
    # Training loop
    for epoch in range(num_epochs):
        average_train_loss = one_epoch_train(model, train_loader, device, loss_fn, optimizer)
        average_test_loss = one_epoch_val(model, test_loader, device, loss_fn)
        
        train_hist.append(average_train_loss)
        test_hist.append(average_test_loss)
        
            
        print(f'Epoch [{epoch+1}/{num_epochs}] - Training Loss: {average_train_loss:.4f} - Test Loss: {average_test_loss:.4f}')

    return train_hist, test_hist

def final_train(num_epochs, model, loader, device, loss_fn, optimizer):
    train_hist =[]
    for epoch in range(num_epochs):
        average_train_loss = one_epoch_train(model, loader, device, loss_fn, optimizer)
        train_hist.append(average_train_loss)   
        print(f'Epoch [{epoch+1}/{num_epochs}] - Training Loss: {average_train_loss:.4f}')
    return train_hist