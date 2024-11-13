import subprocess
import os


def install_llama_cpp():
    try:
        subprocess.call('CMAKE_ARGS=-DLLAMA_CUBLAS=on FORCE_CMAKE=1 pip install --force-reinstall --no-cache-dir --upgrade llama-cpp-python==0.2.23', shell=True)
        print("Successfully installed llama-cpp-python with CUBLAS support.")
    except subprocess.CalledProcessError as e:
        print("Failed to install llama-cpp-python:", e)
        sys.exit(1)
        
def main_wrapper():
    from inference import main
    main()
        
if __name__=="__main__":
    install_llama_cpp()
    main_wrapper()
    