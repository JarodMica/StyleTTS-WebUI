# StyleTTS WebUI
An all-in-one inferencing and training WebUI for StyleTTS.  The intended compatbility is meant for Windows, but should still work with a little bit of modification for WSL or Linux.
> StyleTTS actually trains nicer in WSL than windows, so I might add compatibiltiy here sometime in the future.

## Features
✔️ Inferencing/Generation Tab with ability to choose between different trained models

✔️ Dataset prepration using Whisperx

✔️ Training tab with tensorboard monitoring available

## YouTube Video
Tutorial and installation here: https://youtu.be/dCmAbcJ5v5k

## Setup
There is no Linux or Mac set-up at the moment. However, I think the set-up on linux isn't too convoluted as it doesn't require any code modifications, just installation modifications.  I believe you do not need to uninstall and reinstall torch and then the back slashes should be replaced with forward slashes in the commands below.

### Windows Package
Is available for Youtube Channel Members at the Supporter (Package) level: https://www.youtube.com/channel/UCwNdsF7ZXOlrTKhSoGJPnlQ/join

**Minimum Requirements**
- Nvidia Graphics Card (12GB VRAM is the minimum recommendation for training at a decent speed, 8GB possible though, albeit very slow. See below troubleshooting for more information)
- Windows 10/11
1. After downloading the zip file, unzip it.
2. Launch the webui with launch_webui.bat

### Manual Installation (Windows only)
**Prerequisites**
- Python 3.11: https://www.python.org/downloads/release/python-3119/
- git cmd tool: https://git-scm.com/
- vscode or some other IDE (optional)
- Nvidia Graphics Card (12GB VRAM is the minimum recommendation for training at a decent speed, 8GB possible though, albeit very slow. See below troubleshooting for more information)
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
5. Run the requirements.txt (Before this, make sure you have microsoft build tools installed, else, it will fail for some packages)
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

## Usage
There are 3 Tabs: Generation, Training, and Settings

### Generation
Before you start generating, you need a small reference audio file (preferably wave file) to generate style vectors from.  This can be used for "zero shot" cloning as well, but you'll do the same thing for generating after training a model.

To do this, go into the ```voices``` folder, then create a new folder and name it whatever speaker name you'd like.  Then, place the small reference audio file into that folder.  The full path should look like below:
```
voices/name_of_your_speaker/reference_audio.wav
```
If you had already launched the webui, click on the ```Update Voices``` button and it'll update the voices that are now available to choose from.

One thing to note is the ```Settings``` tab contains the StyleTTS models that are available, but by default, if no training has been done, the base pretrained model will be selected.  After training, you'll be able to change what model is loaded.

|Field      |Description|
|-----------|-----------|
|Input text| The text you want to generate |
|Voice| Voices that are available |
|Reference Audio| The audio file to use as a reference for generation|
|Seed| A number randomly assigned to each generation.  A seed will generate the same audio output no matter how many times you generate.  Set to -1 to have it be randomized|
|alpha| Affects speaker timbre, the higher the value, the further it is from the reference sample. At 0, may sound closer to reference sample at the cost of a little quality|
|beta| Affects speaker prosody and expressiveness.  The higher the value, the more exaggerated speech may be.|
|Diffusion Steps| Affects quality at the cost of some speed.  The higher the number, the more denoising-steps are done (in relation to diffusion models not audio noise)|
|Embedding Scale| Affects speaker expressiveness/emotion.  A higher value may result in higher emotion or expression.|

### Training
Please check the related YouTube video here: https://youtu.be/dCmAbcJ5v5k, start from around 13:05

## Troubleshooting 
Check either installation or running down below in case you run into some issues.  ALL ISSUES may not be covered, I'm bound to miss somethings,

### You have 8GB of VRAM?
It should be possible to train, but data will overflow onto CPU RAM (making training slower by a lot). At these settings, I was clocking in at 8.5GB of VRAM usage:
  - Batch Size = 1
  - Max Length = 100
  - Diffusion Epoch = Set a number higher than Epochs (disables this training)
  - Join Epoch = Set a number higher than Epochs (disables this training)

You may be in luck though because 10-20 epochs of finetuning may be all you need for something decent.  Set it, then go do something else for 24 hours.  Max Length below 100 will cause issues, you can try it, but I didn't get anything good out of it.

### Installation
I reckon there will be a lot of errors that I have either come across or not.  If you have the packaged version, you shouldn't have to troubleshoot. If you do run into software issues, I will address them directly; difficulties in using the software are not included.  

Here are some that I came across:
1. **OSError: [WinError 1314] A required privilege is not held by the client:**
    - Occurs after transcribing for the first time after downloading whisper model.  Just re-run the process and it should work out fine

2. **cudnn or cublas .dll files are not found**
    - Ensure you're using torch 2.3.1 as shown above

3. **Error processing file '/usr/share/espeak-ng-data\phontab': No such file or directory.**
    - eSpeak-NG not installed on your device, see above installation instructions
    - Check: https://github.com/JarodMica/StyleTTS-WebUI/issues/8#issuecomment-2294998032

### Running StyleTTS2
1. ```torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate xx.xx MiB. GPU```
    - Your GPU doesn't have enough VRAM for the configurations you saved for training a voice.  Lower batch size to 1, try again (may cause issue notied in 2).  If not, then lower Max Length in intervals of 50 till it either works or reachs 100 for Max Length.
    - If you hit 100 for Max Length and you still run into issues, set "Diffusion Epoch" and "Joint Epoch" to values that are higher than what you set "Epochs" to.  This disables diffusion and joint training, but the output quality on inference (generation) might suffer.
    - There's a discussion here that talks more about these settings: https://github.com/yl4579/StyleTTS2/discussions/81
2. ```RuntimeError: CUDA error: an illegal memory access was encountered``` OR ```RuntimeError: GET was unable to find an engine to execute this computation```
    - Running with batch size of 1 and max length might be too high even if GPU isn't fully saturated with data.  Not entirely sure why this happens, but try to keep batch size at 2.  Batch size of 1 may allow you to train with longer max_length, but that's when I see this error happen the most.
    - This does NOT occur while training in wsl/linux as far as I've tested
3. **Training is VERY slow**
    - Open task manager and check how much VRAM is being used by going to the performance tab and clicking on GPU.  If you notice that "Dedicated GPU memory" is full, and that "GPU memory" usage is higher than "Dedicated GPU memory" or "Shared GPU memory" is being used, training data is overflowing onto your CPU RAM which will severly hurt training speeds.
    - Two things:
      1. Your GPU cannot handle the bare minimum training requirements for StyleTTS2, there's no solution other than upgrading to more VRAM.
      2. Continue training, just at the slower rate.
        - It should finish, but may take 2-10x the time that it would normally take if you could fit it all into VRAM
4. **FileNotFoundError: [Errno 2] No such file or directory: 'training/name_of_voice/train_phoneme.txt'**
    - You didn't run the ```Run Phonemization``` button after ```Transcribe and Process```, OR something went wrong during that process.
5. **Training Error: ZeroDivisionError: division by zero**
    - Not enough train files inside validation_phoneme.txt, you'll need more data or check the below issue here: https://github.com/JarodMica/StyleTTS-WebUI/issues/14
  
## Continuous Development
I don't have plans for active development of this project - after fixing some bugs, it will most likely end up in a state of dormancy until something else breaks it (package dependencies), or a feature is heavily requested to be added.   

My projects often spring from bursts of motivation surrounding certain tools at certain times.  If I'm no longer using a tool actively, my development of it will reflect this as I can only realistically maintain and develop tools that I actually need to or want to use.  

## Acknowledgements
Huge thanks to the developers responsible for developing StyleTTS2: https://github.com/yl4579/StyleTTS2

## Usage Notice
The base pre-trained StyleTTS2 model used here comes with a License of:

**Pre-Trained Models:** Before using these pre-trained models, you agree to inform the listeners that the speech samples are synthesized by the pre-trained models, unless you have the permission to use the voice you synthesize. That is, you agree to only use voices whose speakers grant the permission to have their voice cloned, either directly or by license before making synthesized voices public, or you have to publicly announce that these voices are synthesized if you do not have the permission to use these voices.
