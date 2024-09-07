@echo off

:: Check if the runtime directory exists
if exist runtime (
    echo "Runtime directory found. Proceeding with TensorBoard in runtime..."

    :: Try to run TensorBoard
    echo Attempting to start TensorBoard...
    runtime\Scripts\tensorboard.exe --logdir training

    :: Check if the previous command failed
    if %errorlevel% equ 0 (
        echo TensorBoard failed to start, attempting to reinstall TensorBoard...
        
        :: Force uninstall TensorBoard
        runtime\python.exe -m pip uninstall -y tensorboard
        
        :: Reinstall TensorBoard
        runtime\python.exe -m pip install tensorboard
        
        :: Try running TensorBoard again
        runtime\Scripts\tensorboard.exe --logdir training
    )
) else (
    echo "Runtime directory not found. Activating virtual environment and running TensorBoard..."
    
    :: Activate the virtual environment and run TensorBoard
    call venv\Scripts\activate
    tensorboard --logdir training
)

pause
