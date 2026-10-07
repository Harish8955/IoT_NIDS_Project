@echo off
echo =========================================================
echo       STARTING BATCH RUN: LOCO WORMS AND SHELLCODE
echo =========================================================

echo.
echo --- [1/6] Running Worms LOCO - 50 Epochs ---
python train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model dual-engine --hide_class Worms --epochs 50 --balance
if %errorlevel% neq 0 ( echo ERROR in Run 1 & exit /b %errorlevel% )

echo.
echo --- [2/6] Running Worms LOCO - 100 Epochs ---
python train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model dual-engine --hide_class Worms --epochs 100 --balance
if %errorlevel% neq 0 ( echo ERROR in Run 2 & exit /b %errorlevel% )

echo.
echo --- [3/6] Running Worms LOCO - 200 Epochs ---
python train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model dual-engine --hide_class Worms --epochs 200 --balance
if %errorlevel% neq 0 ( echo ERROR in Run 3 & exit /b %errorlevel% )

echo.
echo --- [4/6] Running Shellcode LOCO - 50 Epochs ---
python train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model dual-engine --hide_class Shellcode --epochs 50 --balance
if %errorlevel% neq 0 ( echo ERROR in Run 4 & exit /b %errorlevel% )

echo.
echo --- [5/6] Running Shellcode LOCO - 100 Epochs ---
python train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model dual-engine --hide_class Shellcode --epochs 100 --balance
if %errorlevel% neq 0 ( echo ERROR in Run 5 & exit /b %errorlevel% )

echo.
echo --- [6/6] Running Shellcode LOCO - 200 Epochs ---
python train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model dual-engine --hide_class Shellcode --epochs 200 --balance
if %errorlevel% neq 0 ( echo ERROR in Run 6 & exit /b %errorlevel% )

echo.
echo =========================================================
echo       ALL 6 EXPERIMENTS COMPLETED SUCCESSFULLY!
echo =========================================================
pause