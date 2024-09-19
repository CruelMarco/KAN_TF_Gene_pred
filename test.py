from kan import *
from kan.utils import create_dataset
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# initialize KAN with G=3
#model = KAN(width=[642,1,1], grid=3, k=3, seed=1)

# create dataset
f = lambda x: torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]]**2)
dataset = create_dataset(f, n_var=642)

# Fit the model

lamb = [0 , 0.01, 0.001 , 0.0001 , 0.00001 , 0.000001]

for i in lamb:
    
    model = KAN(width=[642,1,1], grid=3, k=3, seed=1)
    
    results = model.fit(dataset, opt="LBFGS", steps=10, lamb=i)
    
    # Assuming results contains 'train_loss' and 'test_loss' lists
    epochs = list(range(1, 11))  # Generating a list of epochs from 1 to 50
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, results['train_loss'], label='Train Loss')
    plt.plot(epochs, results['test_loss'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Testing Loss Over Epochs for Lambda= {i}')
    plt.legend()
    plt.show()
