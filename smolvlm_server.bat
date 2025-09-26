@echo off

cd C:\Users\lmgre\Documents\SIU\Senior Design\landon_repo\llama.cpp\build\bin\Release

call .\llama-server.exe -m "C:\Users\lmgre\Documents\SIU\Senior Design\landon_repo\models\SmolVLM-500M-Instruct-Q8_0.gguf" --mmproj "C:\Users\lmgre\Documents\SIU\Senior Design\landon_repo\models\mmproj-SmolVLM-500M-Instruct-Q8_0.gguf" -ngl 99

cmd