# StyleTTS WebUI
An all-in-one inferencing and training WebUI for StyleTTS.  The intended compatbility is meant for Windows, but should still work with a little bit of modification for WSL or Linux.
> StyleTTS actually trains nicer in WSL than windows, so I might add compatibiltiy here sometime in the future
## Features
✔️ Inferencing/Generation Tab with ability to choose between different trained models

✔️ Dataset prepration using Whisperx

✔️ Training tab with tensorboard monitoring available

## Setup
### Windows Package
Will be available to YouTube members.  No pre-requisites other than GPU needed

### Manual Installation
**Prerequisites**
- Python 3.11
- git cmd tool:
- vscode or some other IDE
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
```
pip uninstall torch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
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






