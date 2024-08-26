import torch
from torch.utils.data import Dataset, DataLoader
from agents.MinimaxAgent import MinimaxAgent
import torch.nn as nn
import torch.optim as optim

from environments.reservoir.Mean import MeanReservoirGameEnvironment
from environments.reservoir.ReservoirGameEnvironment import *

device = torch.device("cpu")


class CustomDataset(Dataset):
    def __init__(self, env_type):
        self.inputs = []
        self.targets = []
        game = Game()
        env = env_type(game)
        agent = MinimaxAgent()
        for board_index in range(3 ** game.board.area):
            index = board_index
            game.board.clear()
            count_x = 0
            count_o = 0
            moves = []
            for idx in range(game.board.area):
                piece = index % game.board.size
                index //= game.board.size
                if piece == 1:
                    game.board.board_o |= (1 << idx)
                    count_o += 1
                elif piece == 2:
                    game.board.board_x |= (1 << idx)
                    count_x += 1
            if count_o > count_x or count_o < count_x-1:
                continue
            player = 1 - 2 * ((count_x + count_o) % 2)
            self.inputs.append(env.encoded_board(player))
            for move in range(game.board.area):
                if game.board[move] == 0:
                    est = agent.evaluate_move(move, game, player, 0,
                                              float('-inf'), float('inf'),
                                              False)
                    moves.append((est+1)/2)  # [-1,1] => [0,1]
                else:
                    moves.append(-1)
            self.targets.append(moves)
        self.inputs = torch.tensor(self.inputs).to(device)
        self.targets = torch.tensor(self.targets).to(device)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class FullyConnectedNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FullyConnectedNetwork, self).__init__()
        n1_size = 64
        n2_size = 128
        self.fc1 = nn.Linear(input_dim, n1_size)
        self.fc2 = nn.Linear(n1_size, n2_size)
        self.fc3 = nn.Linear(n2_size, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


dataset = CustomDataset(MeanReservoirGameEnvironment)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
model = FullyConnectedNetwork(input_dim=6, output_dim=9).to(device)
# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 2000
loss = None
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
