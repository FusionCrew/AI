Write-Host "Starting Hesitation Detection Model Training (LSTM)..."
Write-Host "Using Python from venv..."

# 가상환경의 Python 실행 파일 사용
$PYTHON_EXE = ".\venv\Scripts\python.exe"

if (-not (Test-Path $PYTHON_EXE)) {
    Write-Error "Virtual environment not found at .\venv"
    exit 1
}

# 학습 실행
& $PYTHON_EXE -m hesitationLearning.run_training --binary

if ($LASTEXITCODE -ne 0) {
    Write-Host "Training failed with exit code $LASTEXITCODE" -ForegroundColor Red
} else {
    Write-Host "Training completed successfully." -ForegroundColor Green
}

Pause
