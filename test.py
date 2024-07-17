import os

import torch

# Set environment variable to avoid OpenMP initialization conflict
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


print(torch.__version__)
print(torch.cuda.is_available())
