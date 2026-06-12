@echo off
echo ========================================
echo  RAG QA System - Environment Setup
echo ========================================

set PROJECT_DIR=D:\projects_with_GPT\LLM-Based_Q&A
set VENV_PYTHON=%PROJECT_DIR%\.venv\Scripts\python.exe
set VENV_PIP=%PROJECT_DIR%\.venv\Scripts\pip.exe

echo.
echo [1/3] Upgrading pip...
"%VENV_PYTHON%" -m pip install --upgrade pip

echo.
echo [2/3] Installing all dependencies from requirements.txt...
"%VENV_PIP%" install -r "%PROJECT_DIR%\requirements.txt"

echo.
echo [3/3] Verifying key packages...
"%VENV_PYTHON%" -c "import langchain; print('langchain OK:', langchain.__version__)"
"%VENV_PYTHON%" -c "import chromadb; print('chromadb OK:', chromadb.__version__)"
"%VENV_PYTHON%" -c "import sentence_transformers; print('sentence-transformers OK:', sentence_transformers.__version__)"
"%VENV_PYTHON%" -c "import fastapi; print('fastapi OK:', fastapi.__version__)"
"%VENV_PYTHON%" -c "import fitz; print('pymupdf OK:', fitz.__version__)"
"%VENV_PYTHON%" -c "import ragas; print('ragas OK:', ragas.__version__)"

echo.
echo ========================================
echo  Setup complete! Activate venv with:
echo  .venv\Scripts\activate.bat
echo ========================================
pause
