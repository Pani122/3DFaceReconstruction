import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

from functools import reduce
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
import vrn_unguided
torch.cuda.empty_cache()

class ImagePairsDataset(Dataset):
    def __init__(self, input_folder, output_folder, transform=None):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.transform = transform

        # Get a list of file names in both input and output folders
        self.input_files = sorted(os.listdir(input_folder))
        self.output_files = sorted(os.listdir(output_folder))

        # Assuming that input and output file names match, e.g., input_21.jpg and output_21.jpg
        self.pair_files = [(output_name.replace('.npy', '.jpg'), output_name) for output_name in self.output_files if self.has_corresponding_output(output_name)]
        print(len(self.pair_files))
    def has_corresponding_output(self, output_name):
        # Check if there is a corresponding output file for the given input file
        input_name = output_name.replace('.npy', '.jpg')
        return input_name in self.input_files
    def __len__(self):
        return len(self.pair_files)

    def __getitem__(self, index):
        input_name, output_name = self.pair_files[index]

        # Load input and output images
        input_path = os.path.join(self.input_folder, input_name)
        output_path = os.path.join(self.output_folder, output_name)

        input_image = Image.open(input_path).convert("RGB")
        output_matrix = np.load(output_path)

        output_tensor = torch.from_numpy(output_matrix).float()

        if self.transform:
            input_image = self.transform(input_image)
        
        return input_image, output_tensor


transform = transforms.Compose([transforms.Resize((192, 192)), transforms.ToTensor()])

input_folder = "inputs"
output_folder = "outputs"

dataset = ImagePairsDataset(input_folder, output_folder, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Initialize your model, loss function, and optimizer
model = vrn_unguided
criterion = nn.CrossEntropyLoss()  # Replace with your actual loss function
optimizer = optim.RMSprop(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        #print(targets.shape)
        # Forward pass
        outputs = model(inputs)[0]

        # Compute the loss
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print training statistics
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item()}")
        torch.cuda.empty_cache()

# Save the trained model
torch.save(model.state_dict(), 'vrn_unguided.pth')