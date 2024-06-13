import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# Initial data
first_names = ["Ali", "Zahra", "Reza", "Sara", "Mohammad", "Fatemeh", "Hossein", "Maryam", "Mehdi", "Narges", "Hamed", "Roya"]
last_names  = ["Ahmadi", "Hosseini", "Karimi", "Rahimi", "Hashemi", "Ebrahimi", "Moradi", "Mohammadi", "Rostami", "Fazeli", "Hosseinzade", "Niknam"]

# Merging names and lastnames
guests = [f"{fn} {ln}" for fn in first_names for ln in last_names]
random.shuffle(guests)
guests = guests[:24]

# Generating dislike matrix
dislike = np.zeros((24, 24), dtype=int)
positions = np.random.choice(576, 40, replace=False)
np.put(dislike, positions, 1)

# Convert dislike matrix to conflict list format
list_of_conflicts = []
for _ in range(500):
    pairs_list = set()
    matrix = np.zeros((24, 24), dtype=int)
    
    while matrix.sum() < 40:
        num1 = np.random.choice(range(24))
        num2 = np.random.choice(range(24))
        
        if num1 == num2:
            continue
        
        pair = (num1, num2)
        if pair in pairs_list:
            continue
        
        pairs_list.add(pair)
        matrix[num1, num2] = 1

    if not any(np.array_equal(matrix, conflict) for conflict in list_of_conflicts):
        list_of_conflicts.append(matrix)

# Convert list of conflicts to a numpy array
conflicts = np.array(list_of_conflicts)

# Function to create adjacent mask
def create_adjacent_mask(n_seats, seats_per_row, seats_per_col):
    adjacent_mask = np.zeros((n_seats, n_seats))
    for i in range(n_seats):
        if i % seats_per_row != 0:
            adjacent_mask[i, i-1] = 1
        if i % seats_per_row != seats_per_row - 1:
            adjacent_mask[i, i + 1] = 1
        if i >= seats_per_row:
            adjacent_mask[i, i - seats_per_row] = 1
        if i < n_seats - seats_per_row:
            adjacent_mask[i, i + seats_per_row] = 1
    return adjacent_mask

adjacent_mask = create_adjacent_mask(24, 6, 4)
adjacent_mask = torch.tensor(adjacent_mask, dtype=torch.float32)

# Define a function to calculate total conflicts for a given arrangement
def calculate_conflict(seating_arrangement, conflict_matrix):
    ca_mul = conflict_matrix * adjacent_mask
    conflicts = torch.sum(torch.matmul(seating_arrangement.float(), ca_mul))
    return conflicts

# Custom loss function using the conflict calculation
def custom_loss(predicted_seating_arrangement, conflicts_tensor):
    loss = 0
    for i in range(predicted_seating_arrangement.shape[0]):
        conflict = calculate_conflict(predicted_seating_arrangement[i], conflicts_tensor[i])
        occurrences = torch.sum(predicted_seating_arrangement[i], dim=0)
        repetitive_elements = torch.sum(torch.abs(occurrences - 1)) / 24
        unique_penalty = torch.sum(torch.abs(torch.sum(predicted_seating_arrangement[i], dim=1) - 1)) / 24
        loss += unique_penalty + repetitive_elements + conflict
    return loss / predicted_seating_arrangement.shape[0]

conflicts_tensor = torch.tensor(conflicts, dtype=torch.float32)

# Define the neural network
class SeatingArrangementModel(nn.Module):
    def __init__(self):
        super(SeatingArrangementModel, self).__init__()
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(24*24, 10)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(10, 24*24)
        self.softmax = nn.Softmax(dim=1)
        self.reshape = lambda x: x.view(-1, 24, 24)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.softmax(x)
        x = self.reshape(x)
        return x

model = SeatingArrangementModel()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Function to enforce unique guest assignments
def enforce_unique_guests(arrangement):
    n = arrangement.size
    flat_arrangement = arrangement.flatten()
    unique_elements, counts = np.unique(flat_arrangement, return_counts=True)
    duplicates = unique_elements[counts > 1]
    missing_elements = [i for i in range(n) if i not in flat_arrangement]
    
    for dup in duplicates:
        duplicate_indices = np.where(flat_arrangement == dup)[0]
        for idx in duplicate_indices[1:]:
            if missing_elements:
                flat_arrangement[idx] = missing_elements.pop(0)
            else:
                break
    
    return flat_arrangement.reshape(arrangement.shape)

# Training loop
for epoch in range(2):
    model.train()
    optimizer.zero_grad()
    predicted_seating_arrangement = model(conflicts_tensor)
    loss = custom_loss(predicted_seating_arrangement, conflicts_tensor)
    loss.backward()
    optimizer.step()
    print(f'Epoch: {epoch}, Loss: {loss.item()}')

    # Convert the first prediction to a seating arrangement of 6x4 and print the arrangement
    with torch.no_grad():
        first_seating_arrangement = torch.argmax(predicted_seating_arrangement[0], dim=1).view(6, 4)
        arrangement = enforce_unique_guests(first_seating_arrangement.cpu().numpy())
        arranged_guests = np.array(guests)[arrangement]
        print(f'Arrangement for epoch {epoch}:\n', arranged_guests)

# Get final predictions
model.eval()
with torch.no_grad():
    predicted_seating_arrangement = model(conflicts_tensor)

# Convert the final prediction to a seating arrangement of 6x4
final_seating_arrangement = torch.argmax(predicted_seating_arrangement[0], dim=1).view(6, 4)
arrangement = enforce_unique_guests(final_seating_arrangement.cpu().numpy())
arranged_guests = np.array(guests)[arrangement]
print('Final Arrangement:\n', arranged_guests)
