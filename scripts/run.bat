@echo off
setlocal enabledelayedexpansion

for %%i in (*.inp) do (
    set "filename=%%~ni"

    del /f /q "!filename!.odb" "!filename!.log" "!filename!.lck" "!filename!.com" "!filename!.dat" "!filename!.prt" "!filename!.sta" "!filename!.msg" "!filename!.res"

    abaqus job=!filename! cpus=4 int
)

echo All jobs finished.
pause
