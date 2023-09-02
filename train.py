import torch
import torchvision.transforms as transforms
from DDT.model import get_model
from DDT.ddt import ddt
from Data.data_prepare import get_dataloaders
from ControlNet.pipeline import inpaint
import yaml

# get coefficient model from model.py in DDT ✅

# get dataset from data_prepare.py in Data ✅
# training loop ✅

# run the model and generate coefficients ✅

# get conditions from ddt.py in DDT ✅

# inpaint with pipeline.py in ControlNet ✅

# Loss on problem source

# Loss on coefficient generator ✅


def main():
    # Read the necessary parameters from the config file
    with open("config.yml", 'r') as file:
        conf = yaml.safe_load(file)["model"]
    # Hyperparameters
    learning_rate = conf["learning_rate"]
    num_epochs = conf["num_epochs"]

    # Initialize your model
    model = get_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    # Initialize your dataset
    train_loader, val_loader, test_loader = get_dataloaders()

    transform = transforms.ToTensor()

    # Training loop
    for epoch in range(num_epochs):
        for batch_data in train_loader:
            distorted_images = batch_data['distorted'][0]
            mask_images = batch_data['mask'][0]
            reference_images = batch_data['reference'][0]
            flawless_images = batch_data['flawless'][0]

            # TODO get data from batch and process on them one by one

            # concat distorted image and corresponding mask
            ddt_input = torch.cat((transform(distorted_images), transform(mask_images)), dim=0)
            ddt_input = torch.stack([ddt_input])
            # get alpha
            alpha = model(ddt_input)
            # get conditions
            conditions = ddt(distorted_images, reference_images)

            # Inpainting
            inpaint_result = inpaint(distorted_images, mask_images, conditions)

            # TODO new loss to find the cause of problem (Inpainting model or Coefficient generator)

            # Calculate the loss
            loss = criterion(inpaint_result, flawless_images)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print loss for monitoring
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

    # Save the trained model
    torch.save(model.state_dict(), 'trained_model.pth')


if __name__ == '__main__':
    main()
