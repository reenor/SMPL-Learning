# VIBE: Video Inference for Human Body Pose and Shape Estimation
[Official Project Link](https://github.com/mkocabas/VIBE)

## Running
Launch Anaconda Prompt as administrator

Activate the virtual environment
```Shell
set "PATH_TO_VENV=C:\venvs\vibe" && conda activate %PATH_TO_VENV%
```
Navigate to VIBE
```Shell
set "PATH_TO_VIBE=D:\Projects\Repos-VIBE\VIBE" && call cd /d %^PATH_TO_VIBE%
```
Execute
```Shell
python demo_windows.py --vid_file ../vibe_input/sample_video.mp4 --output_folder ../vibe_output --display
```

## Windows Installation

### 0. Tools set-up
We need to install the following tools:
- Python: https://www.python.org/downloads/
- GIT: https://git-scm.com/download/win
- MSVC++ Redist (14.40.33810.0) : https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170
- FFMPEG (7.0.1) : https://www.gyan.dev/ffmpeg/builds/#release-builds
> Note: after extracted FFMPEG, add folder `bin` to the `Path` in System Environment Variables

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
set "PATH_TO_VENV=C:\venvs\vibe"
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

### 4. VIBE
For downloading YouTube videos
```Shell
pip install mkl intel-openmp
pip install git+https://github.com/giacaglia/pytube.git --upgrade
```

Install VIBE
```Shell
set "PATH_TO_PROJECT=D:\Projects\Repos-VIBE" && set "PATH_TO_VIBE=D:\Projects\Repos-VIBE\VIBE"
```
```Shell
mkdir %PATH_TO_PROJECT% && cd /d %PATH_TO_PROJECT%
```
```Shell
git clone https://github.com/reenor/VIBE
```
```Shell
cd /d %PATH_TO_VIBE% && pip install -r requirements.txt
```

### 5. Input data

Download trained models and sample video
```Shell
mkdir %PATH_TO_VIBE%\data && cd /d %PATH_TO_VIBE%\data
```
```Shell
gdown "https://drive.google.com/uc?id=1untXhYOLQtpNEy4GTY_0fL_H-k6cTf_r&confirm=t"
```
```Shell
tar -xf vibe_data.zip
```
```Shell
mkdir %PATH_TO_PROJECT%\vibe_input
```
```Shell
move %PATH_TO_VIBE%\data\vibe_data\sample_video.mp4 %PATH_TO_PROJECT%\vibe_input
```

Torch model and yolo config
```Shell
mkdir %HOMEDRIVE%\%HOMEPATH%\.torch\models
```
```Shell
move %PATH_TO_VIBE%\data\vibe_data\yolov3.weights %HOMEDRIVE%\%HOMEPATH%\.torch\models
```
```Shell
mkdir %HOMEDRIVE%\%HOMEPATH%\.torch\config
```
```Shell
move %PATH_TO_VIBE%\yolov3.cfg %HOMEDRIVE%\%HOMEPATH%\.torch\config
```

## References
- https://github.com/carlosedubarreto/vibe_win_install
- CUDA Compute Capability, https://github.com/prabindh/mygpu?tab=readme-ov-file
- Spyder, https://medium.com/@apremgeorge/using-conda-python-environments-with-spyder-ide-and-jupyter-notebooks-in-windows-4e0a905aaac5

