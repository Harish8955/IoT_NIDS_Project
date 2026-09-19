import sys
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def create_manual_pdf(filename="LOCO_ZeroDay_Experiments_Team_Manual.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=35,
        leftMargin=35,
        topMargin=35,
        bottomMargin=35
    )

    styles = getSampleStyleSheet()

    # Custom styles
    primary_color = colors.HexColor("#1e293b")  # Dark slate
    secondary_color = colors.HexColor("#2563eb") # Royal blue
    text_color = colors.HexColor("#334155")
    bg_light = colors.HexColor("#f8fafc")

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=primary_color,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=secondary_color,
        spaceAfter=10
    )

    heading2_style = ParagraphStyle(
        'Heading2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=primary_color,
        spaceBefore=10,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=text_color,
        spaceAfter=6
    )

    code_style = ParagraphStyle(
        'CodeSnippet',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#0f172a"),
        backColor=colors.HexColor("#f1f5f9"),
        borderColor=colors.HexColor("#cbd5e1"),
        borderWidth=0.5,
        borderPadding=5,
        spaceBefore=3,
        spaceAfter=8
    )

    elements = []

    # Title & Header Banner
    elements.append(Paragraph("IoT NIDS Dual-Engine Zero-Day Experiments", title_style))
    elements.append(Paragraph("Direct CLI Command Manual for Team Members (You, Edwin, Anish, Lakshan)", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=secondary_color, spaceAfter=10))

    # Introduction
    intro_text = (
        "This document lists the <b>exact 6 commands</b> assigned to each team member. "
        "Each person runs their 6 commands directly in their terminal after pulling the repo (<code>git pull origin main</code>)."
    )
    elements.append(Paragraph(intro_text, body_style))
    elements.append(Spacer(1, 6))

    # 1. YOU
    elements.append(Paragraph("👤 <b>Your Commands (UNSW-NB15 Dataset - Worms & Shellcode)</b>", heading2_style))
    user_cmds = (
        "# 1. Hide Worms (50 Epochs)<br/>"
        "python train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model dual-engine --hide_class Worms --epochs 50 --balance<br/>"
        "# 2. Hide Worms (100 Epochs)<br/>"
        "python train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model dual-engine --hide_class Worms --epochs 100 --balance<br/>"
        "# 3. Hide Worms (200 Epochs)<br/>"
        "python train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model dual-engine --hide_class Worms --epochs 200 --balance<br/>"
        "# 4. Hide Shellcode (50 Epochs)<br/>"
        "python train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model dual-engine --hide_class Shellcode --epochs 50 --balance<br/>"
        "# 5. Hide Shellcode (100 Epochs)<br/>"
        "python train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model dual-engine --hide_class Shellcode --epochs 100 --balance<br/>"
        "# 6. Hide Shellcode (200 Epochs)<br/>"
        "python train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model dual-engine --hide_class Shellcode --epochs 200 --balance"
    )
    elements.append(Paragraph(user_cmds, code_style))

    # 2. EDWIN
    elements.append(Paragraph("👤 <b>Edwin's Commands (CSE-CIC-IDS2018 Dataset - Infiltration & Bot)</b>", heading2_style))
    edwin_cmds = (
        "# 1. Hide Infiltration (50 Epochs)<br/>"
        "python train.py --dataset data/CIC_IDS2018_combined.csv --dataset_name CIC-IDS2018 --target_col Label --model dual-engine --hide_class Infiltration --epochs 50 --balance<br/>"
        "# 2. Hide Infiltration (100 Epochs)<br/>"
        "python train.py --dataset data/CIC_IDS2018_combined.csv --dataset_name CIC-IDS2018 --target_col Label --model dual-engine --hide_class Infiltration --epochs 100 --balance<br/>"
        "# 3. Hide Infiltration (200 Epochs)<br/>"
        "python train.py --dataset data/CIC_IDS2018_combined.csv --dataset_name CIC-IDS2018 --target_col Label --model dual-engine --hide_class Infiltration --epochs 200 --balance<br/>"
        "# 4. Hide Bot (50 Epochs)<br/>"
        "python train.py --dataset data/CIC_IDS2018_combined.csv --dataset_name CIC-IDS2018 --target_col Label --model dual-engine --hide_class Bot --epochs 50 --balance<br/>"
        "# 5. Hide Bot (100 Epochs)<br/>"
        "python train.py --dataset data/CIC_IDS2018_combined.csv --dataset_name CIC-IDS2018 --target_col Label --model dual-engine --hide_class Bot --epochs 100 --balance<br/>"
        "# 6. Hide Bot (200 Epochs)<br/>"
        "python train.py --dataset data/CIC_IDS2018_combined.csv --dataset_name CIC-IDS2018 --target_col Label --model dual-engine --hide_class Bot --epochs 200 --balance"
    )
    elements.append(Paragraph(edwin_cmds, code_style))

    # 3. ANISH
    elements.append(Paragraph("👤 <b>Anish's Commands (CIC-IOT2023 Dataset - Mirai-greeth & Vulnerability_Scan)</b>", heading2_style))
    anish_cmds = (
        "# 1. Hide Mirai-greeth (50 Epochs)<br/>"
        "python train.py --dataset data/CICIOT23_combined.csv --dataset_name CIC-IOT2023 --target_col label --model dual-engine --hide_class Mirai-greeth --epochs 50 --balance<br/>"
        "# 2. Hide Mirai-greeth (100 Epochs)<br/>"
        "python train.py --dataset data/CICIOT23_combined.csv --dataset_name CIC-IOT2023 --target_col label --model dual-engine --hide_class Mirai-greeth --epochs 100 --balance<br/>"
        "# 3. Hide Mirai-greeth (200 Epochs)<br/>"
        "python train.py --dataset data/CICIOT23_combined.csv --dataset_name CIC-IOT2023 --target_col label --model dual-engine --hide_class Mirai-greeth --epochs 200 --balance<br/>"
        "# 4. Hide Vulnerability_Scan (50 Epochs)<br/>"
        "python train.py --dataset data/CICIOT23_combined.csv --dataset_name CIC-IOT2023 --target_col label --model dual-engine --hide_class Vulnerability_Scan --epochs 50 --balance<br/>"
        "# 5. Hide Vulnerability_Scan (100 Epochs)<br/>"
        "python train.py --dataset data/CICIOT23_combined.csv --dataset_name CIC-IOT2023 --target_col label --model dual-engine --hide_class Vulnerability_Scan --epochs 100 --balance<br/>"
        "# 6. Hide Vulnerability_Scan (200 Epochs)<br/>"
        "python train.py --dataset data/CICIOT23_combined.csv --dataset_name CIC-IOT2023 --target_col label --model dual-engine --hide_class Vulnerability_Scan --epochs 200 --balance"
    )
    elements.append(Paragraph(anish_cmds, code_style))

    # 4. LAKSHAN
    elements.append(Paragraph("👤 <b>Lakshan's Commands (UNSW & CIC-IOT2023 Datasets - Fuzzers & DDoS-ICMP)</b>", heading2_style))
    lakshan_cmds = (
        "# 1. Hide Fuzzers (50 Epochs)<br/>"
        "python train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model dual-engine --hide_class Fuzzers --epochs 50 --balance<br/>"
        "# 2. Hide Fuzzers (100 Epochs)<br/>"
        "python train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model dual-engine --hide_class Fuzzers --epochs 100 --balance<br/>"
        "# 3. Hide Fuzzers (200 Epochs)<br/>"
        "python train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model dual-engine --hide_class Fuzzers --epochs 200 --balance<br/>"
        "# 4. Hide DDoS-ICMP (50 Epochs)<br/>"
        "python train.py --dataset data/CICIOT23_combined.csv --dataset_name CIC-IOT2023 --target_col label --model dual-engine --hide_class DDoS-ICMP --epochs 50 --balance<br/>"
        "# 5. Hide DDoS-ICMP (100 Epochs)<br/>"
        "python train.py --dataset data/CICIOT23_combined.csv --dataset_name CIC-IOT2023 --target_col label --model dual-engine --hide_class DDoS-ICMP --epochs 100 --balance<br/>"
        "# 6. Hide DDoS-ICMP (200 Epochs)<br/>"
        "python train.py --dataset data/CICIOT23_combined.csv --dataset_name CIC-IOT2023 --target_col label --model dual-engine --hide_class DDoS-ICMP --epochs 200 --balance"
    )
    elements.append(Paragraph(lakshan_cmds, code_style))

    # Sync
    elements.append(Paragraph("🔄 <b>Results Sync Command (All Members After Finishing 6 Runs)</b>", heading2_style))
    sync_cmds = (
        "python compile_results.py<br/>"
        "git add results/<br/>"
        "git commit -m \"Add completed 50/100/200 epoch LOCO results\"<br/>"
        "git push origin main"
    )
    elements.append(Paragraph(sync_cmds, code_style))

    elements.append(Spacer(1, 6))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceAfter=6))
    footer_text = "Confidence-Aware Dual-Engine IoT NIDS Project | Sem 3 Computer Networks"
    elements.append(Paragraph(footer_text, ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=colors.HexColor("#94a3b8"), alignment=1)))

    doc.build(elements)
    print(f"[SUCCESS] Successfully generated PDF manual at: {os.path.abspath(filename)}")

if __name__ == "__main__":
    create_manual_pdf()

