# GPU-Enable

# Step 1
PS C:\Users\91956> nvidia-smi
Wed Mar  5 10:41:50 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 571.96                 Driver Version: 571.96         CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3050 ...  WDDM  |   00000000:01:00.0 Off |                  N/A |
| N/A   59C    P8              3W /   60W |       4MiB /   4096MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A           18448    C+G   ...s (x86)\Overwolf\Overwolf.exe      N/A      |
+-----------------------------------------------------------------------------------------+

# Step 2
PS C:\Users\91956> nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Tue_Feb_27_16:28:36_Pacific_Standard_Time_2024
Cuda compilation tools, release 12.4, V12.4.99
Build cuda_12.4.r12.4/compiler.33961263_0

# Step 3
Download & Extract and Copy:
cudnn-windows-x86_64-8.9.7.29_cuda12-archive

# Step 4
pip  install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Step 5
Code:
import torch

print("Number of GPU: ", torch.cuda.device_count())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Step 6

Still it not showing -- GPU in Jupiter Notebook -- Can you pls checkout once.
