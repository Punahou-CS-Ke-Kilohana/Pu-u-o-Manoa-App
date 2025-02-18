import torch
metal = torch.backends.mps.is_available()
cuda = torch.cuda.is_available()
if metal:
    print("Metal is available.")
    if torch.backends.mps.is_built():
        print("Metal is built.")
else:
    print("Metal is not available.")
if cuda:
    print(
        f"Cuda is available at {[torch.cuda.get_device_name(device) for device in range(torch.cuda.device_count())]}."
    )
else:
    print("Cuda is not available.")
