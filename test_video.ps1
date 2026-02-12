param (
    [Parameter(Mandatory=$true)]
    [string]$VideoPath
)

Write-Host "Running Hesitation Detection Test on Video: $VideoPath"
Write-Host "Using Python from venv..."

# 가상환경의 Python 실행 파일 사용
$PYTHON_EXE = ".\venv\Scripts\python.exe"

if (-not (Test-Path $PYTHON_EXE)) {
    Write-Error "Virtual environment not found at .\venv"
    exit 1
}

# 테스트 실행 (따옴표 처리 주의)
& $PYTHON_EXE test_hesitation_video.py --video "$VideoPath"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Test failed with exit code $LASTEXITCODE" -ForegroundColor Red
} else {
    Write-Host "Test completed successfully." -ForegroundColor Green
}

Pause
