import os
import subprocess
import sys

def main():
    # Explicitly use Anaconda Python which has PyTorch CUDA installed for GPU
    conda_python = r"C:\Users\Acer\anaconda3\python.exe"
    if os.path.exists(conda_python):
        py_exec = f'"{conda_python}"'
    else:
        py_exec = f'"{sys.executable}"'

    commands = [
        # 1. Hide Worms (50 Epochs)
        f"{py_exec} train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model dual-engine --hide_class Worms --epochs 50 --balance",
        # 2. Hide Worms (100 Epochs)
        f"{py_exec} train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model dual-engine --hide_class Worms --epochs 100 --balance",
        # 3. Hide Worms (200 Epochs)
        f"{py_exec} train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model dual-engine --hide_class Worms --epochs 200 --balance",

        # 4. Hide Shellcode (50 Epochs)
        f"{py_exec} train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model dual-engine --hide_class Shellcode --epochs 50 --balance",
        # 5. Hide Shellcode (100 Epochs)
        f"{py_exec} train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model dual-engine --hide_class Shellcode --epochs 100 --balance",
        # 6. Hide Shellcode (200 Epochs)
        f"{py_exec} train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model dual-engine --hide_class Shellcode --epochs 200 --balance",
    ]

    print("==========================================================================")
    print("   STARTING RUN OF YOUR 6 LOCO EXPERIMENTS (USING GPU: NVIDIA RTX 2050)")
    print("==========================================================================")

    for idx, cmd in enumerate(commands, 1):
        print(f"\n[{idx}/6] Executing: {cmd}")
        ret = subprocess.run(cmd, shell=True)
        if ret.returncode != 0:
            print(f"❌ Command {idx} failed with exit code {ret.returncode}")

    print("\n==========================================================================")
    print("   ALL YOUR 6 LOCO EXPERIMENTS COMPLETED!")
    print("==========================================================================")

if __name__ == '__main__':
    main()


