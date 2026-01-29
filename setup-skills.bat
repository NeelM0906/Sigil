@echo off
echo.
echo ========================================
echo   Bomboclat Skill Setup
echo ========================================
echo.

REM Get script directory
set SCRIPT_DIR=%~dp0

REM Create skills directory in user's Clawdbot workspace
if not exist "%USERPROFILE%\clawd\skills" (
    echo Creating skills directory...
    mkdir "%USERPROFILE%\clawd\skills"
)

REM Copy unblinded-knowledge skill
echo Installing Unblinded Knowledge skill...
xcopy /E /I /Y "%SCRIPT_DIR%skills\unblinded-knowledge" "%USERPROFILE%\clawd\skills\unblinded-knowledge" >nul

echo.
echo ========================================
echo   ✅ Skill Installed Successfully!
echo ========================================
echo.
echo Location: %USERPROFILE%\clawd\skills\unblinded-knowledge
echo.
echo ========================================
echo   Next Steps:
echo ========================================
echo.
echo 1. Install Python dependencies:
echo    pip install pinecone openai
echo.
echo 2. Set your API keys:
echo    set PINECONE_API_KEY=your-pinecone-key
echo    set OPENAI_API_KEY=your-openai-key
echo.
echo    (Get keys from: https://app.pinecone.io and https://platform.openai.com)
echo.
echo 3. Start Clawdbot:
echo    pnpm clawdbot gateway --port 18789
echo.
echo 4. Activate on WhatsApp:
echo    "activate unblinded knowledge"
echo.
echo ========================================
echo.
pause