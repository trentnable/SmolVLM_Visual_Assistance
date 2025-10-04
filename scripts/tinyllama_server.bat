@echo off

cd C:\Users\lmgre\Documents\SIU\Senior Design\landon_repo\llama.cpp\build\bin\Release

call .\llama-server.exe -m "C:\Users\lmgre\Documents\SIU\Senior Design\landon_repo\models\tinyllama0.3_Q4_K_M.gguf" --port 8081 -ngl 99

cmd