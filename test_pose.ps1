Write-Host "Running Body Gesture (Pose) Test..."
Write-Host "Using Python from venv..."

# 가상환경의 Python 실행 파일 사용
$PYTHON_EXE = ".\venv\Scripts\python.exe"

if (-not (Test-Path $PYTHON_EXE)) {
    Write-Error "Virtual environment not found at .\venv"
    exit 1
}

# 테스트 실행 (새로운 경로)
# test.py는 내부적으로 sys.path를 추가하므로 바로 실행 가능
& $PYTHON_EXE hesitationLearning/pose/test.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "Test failed with exit code $LASTEXITCODE" -ForegroundColor Red
} else {
    Write-Host "Test completed successfully." -ForegroundColor Green
}

Pause
