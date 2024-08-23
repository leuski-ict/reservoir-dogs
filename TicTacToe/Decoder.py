import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np

from environments.ReservoirGameEnvironment import CONDUCTANCE_TABLE, \
    CONDUCTANCE_TABLE_MAX, CONDUCTANCE_TABLE_MIN


def encode_bits_conductance(conductance):
    sample = np.random.normal(conductance[0], conductance[1])
    return max(min(sample, CONDUCTANCE_TABLE_MAX),
               CONDUCTANCE_TABLE_MIN)


def encode_bits(bit_values):
    conductance = CONDUCTANCE_TABLE[tuple(bit_values)]
    return encode_bits_conductance(conductance)


def bits_from_number(value, size):
    bit_values = []
    bit_count = 0
    for bit_index in range(size):
        bit = value % 2
        if bit == 1:
            bit_count += 1
        bit_values.append(bit)
        value //= 2
    bit_values.reverse()
    with_parity = bit_values.copy()
    with_parity.append(bit_count % 2)
    return bit_values, with_parity


class CustomDataset(Dataset):
    def __init__(self):
        self.inputs = []
        self.targets = []
        for position in range(8):
            bit_values = bits_from_number(position, 3)
            conductance = CONDUCTANCE_TABLE[tuple(bit_values[1])]
            target = [float(x) for x in bit_values[0]]
            for sample_index in range(100):
                sample = encode_bits_conductance(conductance)
                self.inputs.append([sample])
                self.targets.append(target)

        self.inputs = torch.tensor(self.inputs)
        self.targets = torch.tensor(self.targets)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class FullyConnectedNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FullyConnectedNetwork, self).__init__()
        n1_size = 32
        n2_size = 64
        self.fc1 = nn.Linear(input_dim, n1_size)
        self.fc2 = nn.Linear(n1_size, n2_size)
        self.fc3 = nn.Linear(n2_size, output_dim)

    def forward(self, x):
        x = torch.rrelu(self.fc1(x))
        x = torch.rrelu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
model = FullyConnectedNetwork(input_dim=1, output_dim=3)
# Define loss function and optimizer
#criterion = nn.MSELoss()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 2000
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

model.eval()


def test():
    with torch.no_grad():
        for position in range(8):
            bit_values = bits_from_number(position, 3)
            input = [encode_bits(np.array(bit_values[1]))]
            target = bit_values[0]
            x_test = torch.tensor(input, dtype=torch.float32).view(-1, 1)
            bit_predictions = model(x_test)
            # Convert probabilities to binary values
            bit_predictions = (bit_predictions > 0.5).int()
            print(f"{target} Predicted bits: {bit_predictions.numpy()[0]}")


test()
