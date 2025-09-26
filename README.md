#  Visual Assistant

##  Linux Setup Instructions

Clone the Repository

```bash
git clone -b reuben https://github.com/trentnable/SmolVLM_Visual_Assistance.git
```

Move to new directory

```bash
cd SmolVLM_Visual_Assistance\llama.cpp
```

A few of things:
- CUDA is required for the GPU build (can specify specifc version if needed)
    ```bash
    sudo apt install nvidia-cuda-toolkit
- You need CMake
    ```bash
    sudo apt install cmake


- Curl usage is enabled by default and can be turned off with -DLLAMA_CURL=OFF. Otherwise you need to install development libraries for libcurl.

    Debian / Ubuntu:
  
          sudo apt-get install libcurl4-openssl-dev # (or libcurl4-gnutls-dev if you prefer GnuTLS)
  
    Fedora / RHEL / Rocky / Alma:
  
          sudo dnf install libcurl-devel
  
    Arch / Manjaro:
  
          sudo pacman -S curl # includes libcurl headers

Time to build from source. This configuration requires a GPU, and will take a while.

```bash
mkdir build
cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF
cmake --build build --config Release
```

For CPU build:
```bash
mkdir build
cmake -B build
cmake --build build --config Release
```

To test the build, run:
```bash
cd build\bin\Release && .\llama-cli.exe --version
```

Next, you need to make the folder where you will store models used by the llama server.

From the project directory:

```bash
mkdir models && cd models
```

Download the models to the folder

```bash
curl -L -o mmproj-SmolVLM-500M-Instruct-Q8_0.gguf "https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/main/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf"

curl -L -o SmolVLM-500M-Instruct-Q8_0.gguf "https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/main/SmolVLM-500M-Instruct-Q8_0.gguf"

curl -L -o Phi-3-mini-4k-instruct-q4.gguf "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf"

curl -L -o tinyllama0.3_Q4_K_M.gguf "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF/resolve/main/tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf"
```

Edit the 'smolvlm_server.bat', 'tinyllama_server.bat', and 'phi3_server.bat' scripts to use the directories of llama-cli.exe and the models. Templates for the .bat files are included in this repo.

For 'smolvlm_server.bat':
```bash
@echo off

cd project_directory/llama.cpp

./build/bin/llama-server -m "./models/SmolVLM-500M-Instruct-Q8_0.gguf" --mmproj "./project_directory/SmolVLM_Visual_Assistance/models/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf" -ngl 99

```

If you're not using GPU, exclude `-ngl 99`

Run each .bat and it should say it is listening at a port (as you designated in the .bat script). Then double click to lauch the index.html from smolvlm-realtime-webcam folder and it should be working.

# Begin to integrate depth and object detection

Create a python virtual environment

```bash
python -m venv dependencies
```

Edit the 'launch_venv.bat' script to use the directories where your project and virtual environment are.

or run the following command.

```bash
source dependencies/bin/activate
```

##  Credits

- **Model**: [SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct) by HuggingFaceTB  
- **Vision-Language Framework**: [Transformers](https://github.com/huggingface/transformers) by Hugging Face  
- **Script & Integration**: Senior Design Team

---

##  Sources provided by Dr. Anagnostopoulos:  

[Sources](SOURCES.md)







