# SMPL-Learning

## Running
Launch Anaconda Prompt as administrator

Activate the virtual environment
```Shell
set "PATH_TO_VENV=C:\venvs\smpl-learning" && conda activate %PATH_TO_VENV%
```
Navigate to project
```Shell
set "PATH_TO_PROJECT=D:\Projects\SMPL-Learning" && call cd /d %^PATH_TO_PROJECT%
```
Execute
```Shell
python demo.py
```

## Windows Installation

### 0. Tools set-up
We need to install the following tools:
- Python: https://www.python.org/downloads/
- GIT: https://git-scm.com/download/win
- MSVC++ Redist (14.40.33810.0) : https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170
- Miniconda: https://docs.anaconda.com/miniconda/

### 0Bis. CUDA-powered GPU 
First, update the latest GPU driver: https://www.nvidia.com/Download/index.aspx?lang=en-us and restart the system for the changes to take effect.

In order to check our GPU driver, use the following command:
```Shell
nvidia-smi
```
Second, get the CUDA Compute Capability of our GPU, and find out which CUDA versions are supported for this Compute Capability on [CUDA Wikipedia page](https://en.wikipedia.org/wiki/CUDA#GPUs_supported).

For example, GPU on my system is NVIDIA GeForce RTX 3070 -> Compute Capability is **8.6** (or **sm_86**) -> CUDA versions of **11.1 - 11.4** are supported for sm_86. Therefore, we will use this versions when choosing the CUDA runtime (or CUDA Toolkit) in the next steps.

> Note: CUDA version coming from executing command `nvidia-smi` is considered as CUDA Driver version, and it MUST NOT BE USED to install other components, [CUDA Driver vs CUDA Runtime](https://stackoverflow.com/questions/53422407/different-cuda-versions-shown-by-nvcc-and-nvidia-smi)

### 1. Virtual Environment for Python
Launch Anaconda Prompt as administrator

Create a place to store virtual environment
```Shell
set "PATH_TO_VENV=C:\venvs\smpl-learning"
```
Create a virtual environment
```Shell
conda create -p %PATH_TO_VENV% python=3.7
```
Activate the virtual environment
```Shell
conda activate %PATH_TO_VENV%
```
In case there is something wrong
```Shell
conda deactivate && conda remove -p %PATH_TO_VENV% --all
```

### 2. CUDA Toolkit and cuDNN
Version **11.1** will be chose for installing CUDA Toolkit, and cuDNN version 8.1.0 is also selected by searching in [cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive)
```Shell
conda install cudatoolkit=11.1 cudnn=8.1.0 -c conda-forge
```
> Note: cudnn v8.1.0 is compatible with cudatoolkit v11.1

### 3. PyTorch
Based on CUDA **11.1**, install the suitable version of [PyTorch](https://pytorch.org/get-started/previous-versions/)
```Shell
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```

Check if [Torch supports our GPU](https://stackoverflow.com/questions/60987997/why-torch-cuda-is-available-returns-false-even-after-installing-pytorch-with) or not
```
python
>>> import torch
>>> torch.__version__
>>> torch.cuda.is_available()
>>> torch.cuda.get_arch_list()
>>> torch.zeros(1).cuda()
```

### 4. Repository
Clone Repository

```Shell
set "PATH_TO_PROJECT=D:\Projects\SMPL-Learning"
```

```Shell
cd /d %PATH_TO_PROJECT% && pip install -r requirements.txt
```

### 5. Input data

```Shell
mkdir %PATH_TO_PROJECT%\data\models\smpl\origin && mkdir %PATH_TO_PROJECT%\data\models\smpl\removed_chumpy_objects
```
Download pretrained SMPL models at https://smpl.is.tue.mpg.de/download.php and extract model files according to the following structure:

```bash
models
├── smpl
│   ├── origin
|       ├── basicmodel_f_lbs_10_207_0_v1.1.0.pkl
|       ├── basicmodel_m_lbs_10_207_0_v1.1.0.pkl
|       ├── basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl
│   ├── removed_chumpy_objects
├── xxx
```

Run the following to remove any Chumpy objects from the neutral model data
```Shell
python remove_chumpy_objects.py --input-models %PATH_TO_PROJECT%\data\models\smpl\origin\basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl --output-folder %PATH_TO_PROJECT%\data\models\smpl\removed_chumpy_objects
```

## References
- CUDA Compute Capability, https://github.com/prabindh/mygpu?tab=readme-ov-file
- Spyder, https://medium.com/@apremgeorge/using-conda-python-environments-with-spyder-ide-and-jupyter-notebooks-in-windows-4e0a905aaac5

