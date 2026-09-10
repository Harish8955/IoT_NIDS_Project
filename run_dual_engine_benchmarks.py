import subprocess
import sys

def main():
    commands = [
        # UNSW-NB15 Dual-Engine Runs (50, 100, 200 Ep)
        "python train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model dual-engine --hide_class Worms --epochs 50 --balance",
        "python train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model dual-engine --hide_class Worms --epochs 100 --balance",
        "python train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model dual-engine --hide_class Worms --epochs 200 --balance",

        # CSE-CIC-IDS2018 Dual-Engine Runs (50, 100, 200 Ep)
        "python train.py --dataset data/CIC_IDS2018_combined.csv --dataset_name CIC-IDS2018 --target_col Label --model dual-engine --hide_class Infiltration --epochs 50 --balance",
        "python train.py --dataset data/CIC_IDS2018_combined.csv --dataset_name CIC-IDS2018 --target_col Label --model dual-engine --hide_class Infiltration --epochs 100 --balance",
        "python train.py --dataset data/CIC_IDS2018_combined.csv --dataset_name CIC-IDS2018 --target_col Label --model dual-engine --hide_class Infiltration --epochs 200 --balance",

        # CIC-IOT2023 Dual-Engine Runs (50, 100, 200 Ep)
        "python train.py --dataset data/CICIOT23_combined.csv --dataset_name CIC-IOT2023 --target_col label --model dual-engine --hide_class Mirai-greeth --epochs 50 --balance",
        "python train.py --dataset data/CICIOT23_combined.csv --dataset_name CIC-IOT2023 --target_col label --model dual-engine --hide_class Mirai-greeth --epochs 100 --balance",
        "python train.py --dataset data/CICIOT23_combined.csv --dataset_name CIC-IOT2023 --target_col label --model dual-engine --hide_class Mirai-greeth --epochs 200 --balance",
    ]

    print("==========================================================================")
    print(f"   STARTING BATCH RUN OF {len(commands)} DUAL-ENGINE BENCHMARK EXPERIMENTS")
    print("==========================================================================")

    for idx, cmd in enumerate(commands, 1):
        print(f"\n[{idx}/{len(commands)}] Running: {cmd}")
        ret = subprocess.run(cmd, shell=True)
        if ret.returncode != 0:
            print(f"❌ Command failed with exit code {ret.returncode}: {cmd}")

    print("\n==========================================================================")
    print("   ALL DUAL-ENGINE BENCHMARK EXPERIMENTS COMPLETED!")
    print("==========================================================================")

if __name__ == '__main__':
    main()
