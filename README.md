# StyleTTS WebUI
An all-in-one inferencing and training WebUI for StyleTTS.  The intended compatbility is meant for Windows, but should still work with a little bit of modification for WSL or Linux.
> StyleTTS actually trains nicer in WSL than windows, so I might add compatibiltiy here sometime in the future
## Features
✔️ Inferencing/Generation Tab with ability to choose between different trained models

✔️ Dataset prepration using Whisperx

✔️ Training tab with tensorboard monitoring available

## Setup
There is no Linux or Mac set-up at the moment.

### Windows Package
Will be available to YouTube members.  No pre-requisites other than GPU needed

### Manual Installation (Windows only)
**Prerequisites**
- Python 3.11: https://www.python.org/downloads/release/python-3119/
- git cmd tool: https://git-scm.com/
- vscode or some other IDE (optional)
- Nvidia Graphics Card
1. Clone the repository
  ```
git clone https://github.com/JarodMica/StyleTTS-WebUI.git
  ```
2. Navigate into the repo
```
cd .\StyleTTS-WebUI\
```
3. Setup a virtual environement, specifying python 3.11
```
py -3.11 -m venv venv
```
4. Activate venv.  If you've never run venv before on windows powershell, you will need to change ExecutionPolicy to RemoteSigned
```
.\venv\Scripts\activate
```
5. Run the requirements.txt
```
pip install -r .\requirements.txt
```
6. Uninstall and reinstall torch manually as windows does not particularly like just installing torch, you need to install prebuilt wheels.
> **NOTE:** torch installed with 2.4.0 or higher was causing issues with cudnn and cublas dlls not being found (presumed due to ctranslate2).  Make sure you use 2.3.1 as specified in the command below.
```
pip uninstall torch
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```
7. Initialize submodules in the repository
```
git submodule init
git submodule update --remote
```
8. Install the StyleTTS2 package into venv
```
pip install .\modules\StyleTTS2\
```
9. Download the pretrained StyleTTS2 Model and yaml here:https://huggingface.co/yl4579/StyleTTS2-LibriTTS/tree/main/Models/LibriTTS.  You'll need to place them into the folder ```pretrain_base_1``` inside of the ```models``` folder.  The file structure should look like the below.
```
models\pretrain_base_1\epochs_2nd_00020.pth
models\pretrain_base_1\config.yml
```

10. Install eSpeak-NG onto your computer.  Head over to https://github.com/espeak-ng/espeak-ng/releases and select the ```espeak-ng-X64.msi``` the assets dropdown.  Download, run, and follow the prompts to set it up on your device.  As of this write-up, it'll be at the bottom of 1.51 on the github releases page
> You can remove the program by going to "Add or remove programs" on your computer, then searching for espeak.
11. Download punkt by running the below python script:
```
python .\modules\StyleTTS2\styletts2\download_punkt.py
```
12.. Run the StyleTTS2 Webui
```
python webui.py
```








