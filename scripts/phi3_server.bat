@echo off

cd C:\Users\lmgre\Documents\SIU\Senior Design\landon_repo\llama.cpp\build\bin\Release

call .\llama-server.exe -m "C:\Users\lmgre\Documents\SIU\Senior Design\landon_repo\models\Phi-3-mini-4k-instruct-q4.gguf" --port 8082 -ngl 99

cmd