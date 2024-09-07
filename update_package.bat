@echo off
if not exist "runtime" (
    @echo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @echo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    echo This is not the packaged version, if you are trying to update your manual installation, please use git pull instead
    @echo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @echo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    pause
    exit /b
)

if exist "StyleTTS-WebUI" (
    @echo It looks like you've already cloned the repository for updating before.
    @echo If you want to continue with updating, type y.
    @echo Else, type n.
    choice /M "Do you want to continue?"
    if errorlevel 2 (
        @echo Exiting the script...
        exit /b
    )
    rmdir /S /Q "StyleTTS-WebUI"
)

portable_git\bin\git.exe clone https://github.com/JarodMica/StyleTTS-WebUI.git
cd StyleTTS-WebUI
git submodule init
git submodule update --remote
cd ..

xcopy StyleTTS-WebUI\update_package.bat update_package.bat /E /I /H 
xcopy StyleTTS-WebUI\launch_tensorboard.bat launch_tensorboard.bat /E /I /H 
xcopy StyleTTS-WebUI\requirements.txt requirements.txt /E /I /H 

xcopy StyleTTS-WebUI\webui.py webui.py /H
xcopy StyleTTS-WebUI\modules\tortoise_dataset_tools modules\tortoise_dataset_tools /E /I /H
xcopy StyleTTS-WebUI\modules\styletts2_phonemizer modules\styletts2_phonemizer /E /I /H
xcopy StyleTTS-WebUI\modules\StyleTTS2 modules\StyleTTS2 /E /I /H

runtime\python.exe -m pip uninstall StyleTTS2
runtime\python.exe -m pip install modules\StyleTTS2
runtime\python.exe -m pip install -r requirements.txt

@echo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@echo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@echo Finished updating!
@echo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@echo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pause