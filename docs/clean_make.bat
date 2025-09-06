@ECHO OFF

REM clean then make documentation
REM required to delete old files from 'build' folder
REM which may not be updated by make.bat html otherwise

CALL make.bat clean
CALL make.bat html