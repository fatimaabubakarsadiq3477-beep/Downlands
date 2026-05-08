@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"

echo ===============================================
echo Запуск приложения Steam Activation v3.2 Wrapper
echo ===============================================

if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Найдено виртуальное окружение .venv, активируем...
    call ".venv\Scripts\activate.bat"
) else (
    echo [WARN] Виртуальное окружение .venv не найдено.
    echo [WARN] Будет использован системный Python.
)

where streamlit >nul 2>nul
if errorlevel 1 (
    echo [WARN] Команда streamlit не найдена в PATH, пробуем python -m streamlit...
    python -m streamlit --version >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] Streamlit недоступен ни как команда, ни как Python-модуль.
        echo [HINT] Установите зависимости: pip install -r requirements.txt
        pause
        exit /b 1
    )
    echo [INFO] Запуск: python -m streamlit run app.py
    python -m streamlit run app.py
) else (
    echo [INFO] Запуск: streamlit run app.py
    streamlit run app.py
)

if errorlevel 1 (
    echo [ERROR] Streamlit завершился с ошибкой.
    pause
    exit /b 1
)

endlocal
