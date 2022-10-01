import torch
print(f'torch {torch.__version__} {torch.cuda.get_device_properties(0) if torch.cuda.is_available() else "CPU"}') 