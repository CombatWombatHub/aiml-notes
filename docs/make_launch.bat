@ECHO OFF
REM make documentation then launch it in browser window
CALL make.bat html
CALL start build/html/index.html