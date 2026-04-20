@echo off
cd /d "%~dp0"
echo Running Watchlist Seeder...
python seed_watchlist.py
echo.
pause
