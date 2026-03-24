@echo off
setlocal

:: Check if venv exists
if not exist venv (
    echo [INFO] Creating virtual environment...
    python -m venv venv
)

:: Activate venv
echo [INFO] Activating virtual environment...
call venv\Scripts\activate

:: Check for model file
if exist currency_model.pkl goto FOUND
echo.
echo [WARNING] Model file 'currency_model.pkl' not found!
echo Please ensure you have added images to 'archive/data/data/real' and 'archive/data/data/fake' folders.
echo Then run: python train_currency_model.py
echo.
echo Starting web app anyway (predictions will fail until trained)...
goto START

:FOUND
echo [SUCCESS] Model found. Starting web app...

:START
:: Run app
python app.py
pause
