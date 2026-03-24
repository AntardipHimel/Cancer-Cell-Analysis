# =============================================================================
# setup_git.ps1 - Run this ONCE from D:\Research\Cancer_Cell_Analysis\
# =============================================================================
# This script:
#   1. Places .gitkeep in every folder (preserves structure in Git)
#   2. Initializes Git repo
#   3. Stages everything (respecting .gitignore)
#   4. Shows you what will be committed BEFORE committing
# =============================================================================

Write-Host "`n=== OncoLens Git Setup ===" -ForegroundColor Cyan

# Step 1: Drop .gitkeep into every directory
Write-Host "`n[1/3] Creating .gitkeep files in all directories..." -ForegroundColor Yellow

$dirs = Get-ChildItem -Directory -Recurse | Where-Object {
    $_.FullName -notmatch '\\\.git\\' -and
    $_.FullName -notmatch '\\__pycache__\\' -and
    $_.FullName -notmatch '\\\.vscode\\'
}

$count = 0
foreach ($dir in $dirs) {
    $gitkeep = Join-Path $dir.FullName ".gitkeep"
    if (-not (Test-Path $gitkeep)) {
        New-Item -ItemType File -Path $gitkeep -Force | Out-Null
        $count++
    }
}
Write-Host "  Created $count .gitkeep files" -ForegroundColor Green

# Step 2: Initialize git
Write-Host "`n[2/3] Initializing Git repository..." -ForegroundColor Yellow

if (-not (Test-Path ".git")) {
    git init
    Write-Host "  Git repo initialized" -ForegroundColor Green
} else {
    Write-Host "  Git repo already exists" -ForegroundColor Green
}

# Step 3: Stage and show status
Write-Host "`n[3/3] Staging files..." -ForegroundColor Yellow
git add .

Write-Host "`n=== FILES THAT WILL BE COMMITTED ===" -ForegroundColor Cyan
git status

Write-Host "`n=== REVIEW THE LIST ABOVE ===" -ForegroundColor Yellow
Write-Host "Make sure NO datasets, videos, images, or .pt files appear." -ForegroundColor Yellow
Write-Host "`nIf everything looks good, run:" -ForegroundColor Green
Write-Host '  git commit -m "Initial commit: OncoLens cancer cell classification"' -ForegroundColor White
Write-Host '  git branch -M main' -ForegroundColor White
Write-Host '  git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git' -ForegroundColor White
Write-Host '  git push -u origin main' -ForegroundColor White
Write-Host ""
