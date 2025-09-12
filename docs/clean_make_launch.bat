@ECHO OFF
REM clean then make documentation then launch it in browser window
REM necessary if changing certain files (like images) that don't trigger re-build otherwise
REM takes longer to build since it re-runs Jupyter notebooks
CALL make.bat clean
CALL make.bat html
CALL start build/html/index.html