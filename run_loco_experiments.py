import argparse
import subprocess
import sys

# Define LOCO Task Assignment with 50, 100, and 200 Epochs for Each Member (6 Runs per person)
MEMBER_ASSIGNMENTS = {
    "me": [
        # You: UNSW-NB15 ('Worms' & 'Shellcode') @ 50, 100, 200 Epochs
        "python train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model dual-engine --hide_class Worms --epochs 50 --balance",
        "python train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model dual-engine --hide_class Worms --epochs 100 --balance",
        "python train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model dual-engine --hide_class Worms --epochs 200 --balance",

        "python train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model dual-engine --hide_class Shellcode --epochs 50 --balance",
        "python train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model dual-engine --hide_class Shellcode --epochs 100 --balance",
        "python train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model dual-engine --hide_class Shellcode --epochs 200 --balance",
    ],
    "edwin": [
        # Edwin: CSE-CIC-IDS2018 ('Infiltration' & 'Bot') @ 50, 100, 200 Epochs
        "python train.py --dataset data/CIC_IDS2018_combined.csv --dataset_name CIC-IDS2018 --target_col Label --model dual-engine --hide_class Infiltration --epochs 50 --balance",
        "python train.py --dataset data/CIC_IDS2018_combined.csv --dataset_name CIC-IDS2018 --target_col Label --model dual-engine --hide_class Infiltration --epochs 100 --balance",
        "python train.py --dataset data/CIC_IDS2018_combined.csv --dataset_name CIC-IDS2018 --target_col Label --model dual-engine --hide_class Infiltration --epochs 200 --balance",

        "python train.py --dataset data/CIC_IDS2018_combined.csv --dataset_name CIC-IDS2018 --target_col Label --model dual-engine --hide_class Bot --epochs 50 --balance",
        "python train.py --dataset data/CIC_IDS2018_combined.csv --dataset_name CIC-IDS2018 --target_col Label --model dual-engine --hide_class Bot --epochs 100 --balance",
        "python train.py --dataset data/CIC_IDS2018_combined.csv --dataset_name CIC-IDS2018 --target_col Label --model dual-engine --hide_class Bot --epochs 200 --balance",
    ],
    "anish": [
        # Anish: CIC-IOT2023 ('Mirai-greeth' & 'Vulnerability_Scan') @ 50, 100, 200 Epochs
        "python train.py --dataset data/CICIOT23_combined.csv --dataset_name CIC-IOT2023 --target_col label --model dual-engine --hide_class Mirai-greeth --epochs 50 --balance",
        "python train.py --dataset data/CICIOT23_combined.csv --dataset_name CIC-IOT2023 --target_col label --model dual-engine --hide_class Mirai-greeth --epochs 100 --balance",
        "python train.py --dataset data/CICIOT23_combined.csv --dataset_name CIC-IOT2023 --target_col label --model dual-engine --hide_class Mirai-greeth --epochs 200 --balance",

        "python train.py --dataset data/CICIOT23_combined.csv --dataset_name CIC-IOT2023 --target_col label --model dual-engine --hide_class Vulnerability_Scan --epochs 50 --balance",
        "python train.py --dataset data/CICIOT23_combined.csv --dataset_name CIC-IOT2023 --target_col label --model dual-engine --hide_class Vulnerability_Scan --epochs 100 --balance",
        "python train.py --dataset data/CICIOT23_combined.csv --dataset_name CIC-IOT2023 --target_col label --model dual-engine --hide_class Vulnerability_Scan --epochs 200 --balance",
    ],
    "lakshan": [
        # Lakshan: UNSW & CIC-IOT2023 ('Fuzzers' & 'DDoS-ICMP') @ 50, 100, 200 Epochs
        "python train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model dual-engine --hide_class Fuzzers --epochs 50 --balance",
        "python train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model dual-engine --hide_class Fuzzers --epochs 100 --balance",
        "python train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model dual-engine --hide_class Fuzzers --epochs 200 --balance",

        "python train.py --dataset data/CICIOT23_combined.csv --dataset_name CIC-IOT2023 --target_col label --model dual-engine --hide_class DDoS-ICMP --epochs 50 --balance",
        "python train.py --dataset data/CICIOT23_combined.csv --dataset_name CIC-IOT2023 --target_col label --model dual-engine --hide_class DDoS-ICMP --epochs 100 --balance",
        "python train.py --dataset data/CICIOT23_combined.csv --dataset_name CIC-IOT2023 --target_col label --model dual-engine --hide_class DDoS-ICMP --epochs 200 --balance",
    ]
}

def main():
    parser = argparse.ArgumentParser(description="Run LOCO Zero-Day Threat Experiments (50, 100, 200 Epochs) by Team Member")
    parser.add_argument("--member", choices=["me", "edwin", "anish", "lakshan", "all"], default="all",
                        help="Select team member assignment ('me', 'edwin', 'anish', 'lakshan', or 'all')")
    args = parser.parse_args()

    if args.member == "all":
        commands_to_run = []
        for cmds in MEMBER_ASSIGNMENTS.values():
            commands_to_run.extend(cmds)
    else:
        commands_to_run = MEMBER_ASSIGNMENTS[args.member]

    print("==========================================================================")
    print(f"   STARTING LOCO ZERO-DAY EXPERIMENTS FOR ASSIGNMENT: '{args.member.upper()}' ({len(commands_to_run)} runs)")
    print("==========================================================================")

    for idx, cmd in enumerate(commands_to_run, 1):
        print(f"\n[{idx}/{len(commands_to_run)}] Executing: {cmd}")
        ret = subprocess.run(cmd, shell=True)
        if ret.returncode != 0:
            print(f"❌ Command failed with exit code {ret.returncode}")

    print("\n==========================================================================")
    print(f"   LOCO EXPERIMENTS FOR '{args.member.upper()}' COMPLETED!")
    print("==========================================================================")

if __name__ == '__main__':
    main()



