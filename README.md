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
- Nvidia Graphics Card (12gb VRAM is the bare minimum for training at decent speed, see below troubleshooting for more information)
- Microsoft build tools, follow: https://stackoverflow.com/questions/64261546/how-to-solve-error-microsoft-visual-c-14-0-or-greater-is-required-when-inst/64262038#64262038
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
13. (Optional) Make a .bat file to automatically run the webui.py each time without having to activate venv each time. How to: https://www.windowscentral.com/how-create-and-run-batch-file-windows-10
```
call venv\Scripts\activate
python webui.py
```

## Troubleshooting 
Check either installation or running down below in case you run into some issues.  ALL ISSUES may not be covered, I'm bound to miss somethings,

### Installation
I reckon there will be a lot of errors that I have either come across or not.  If you have the packaged version, you shouldn't have to troubleshoot. If you do run into software issues, I will address them directly; difficulties in using the software are not included.  

Here are some that I came across:
1. OSError: [WinError 1314] A required privilege is not held by the client: 
  - Occurs after transcribing for the first time after downloading whisper model.  Just re-run the process and it should work out fine

2. cudnn or cublas .dll files are not found
  - Ensure you're using torch 2.3.1 as shown above

3. Error processing file '/usr/share/espeak-ng-data\phontab': No such file or directory.
  - eSpeak-NG not installed on your device, see above installation instructions

### Running StyleTTS2
1. torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate xx.xx MiB. GPU
  - Your GPU doesn't have enough VRAM for the configurations you saved for training a voice.  Lower batch size to 1, try again.  If not, then lower Max Length in intervals of 50 till it either works or reachs 100 for Max Length.
  - If you hit 100 for Max Length and you still run into issues, set "Diffusion Epoch" and "Joint Epoch" to values that are higher than what you set "Epochs" to.  This disables diffusion and joint training, but the output quality on inference (generation) might suffer.
  - There's a discussion here that talks more about these settings: https://github.com/yl4579/StyleTTS2/discussions/81
2. Training is VERY slow
  - Open task manager and check how much VRAM is being used by going to the performance tab and clicking on GPU.  If you notice that "Dedicated GPU memory" is full, and that "GPU memory" usage is higher than "Dedicated GPU memory" or "Shared GPU memory" is being used, training data is overflowing onto your CPU RAM which will severly hurt training speeds.
  - Two things:
    1. Your GPU cannot handle the bare minimum training requirements for StyleTTS2, there's no solution other than upgrading to more VRAM.
    2. Continue training, just at the slower rate.
      - It should finish, but may take 2-10x the time that it would normally take if you could fit it all into VRAM

