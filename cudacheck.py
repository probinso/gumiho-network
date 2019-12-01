import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0)) # Get name device with ID '0'

# True
