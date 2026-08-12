import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

def build_workload_pdf(filename="Group_Workload_Division_12_11_11_11.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Color Palette - Professional Slate & Teal
    PRIMARY_COLOR = colors.HexColor("#0F172A")    # Deep Slate / Navy
    SECONDARY_COLOR = colors.HexColor("#0D9488")  # Teal Accent
    ACCENT_BLUE = colors.HexColor("#0284C7")      # Sky Blue Highlight
    TEXT_DARK = colors.HexColor("#1E293B")        # High contrast body text
    BG_LIGHT = colors.HexColor("#F8FAFC")         # Table light background
    BORDER_COLOR = colors.HexColor("#CBD5E1")     # Slate border
    ALERT_BG = colors.HexColor("#F0FDFA")         # Light teal card fill

    # Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=PRIMARY_COLOR,
        alignment=TA_CENTER,
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=SECONDARY_COLOR,
        alignment=TA_CENTER,
        spaceAfter=12
    )

    h1_style = ParagraphStyle(
        'Heading1Custom',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=PRIMARY_COLOR,
        spaceBefore=10,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'BodyCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=TEXT_DARK,
        alignment=TA_JUSTIFY,
        spaceAfter=4
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=11.5,
        textColor=colors.white,
        alignment=TA_CENTER
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=TEXT_DARK,
        alignment=TA_LEFT
    )

    story = []

    # =========================================================================
    # HEADER & TITLE BLOCK
    # =========================================================================
    story.append(Paragraph("IoT Network Intrusion Detection System", title_style))
    story.append(Paragraph("Paper Replication Matrix Guide (12 - 11 - 11 - 11 Division)", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=SECONDARY_COLOR, spaceAfter=8))

    # Meta Info Box
    meta_data = [
        [
            Paragraph("<b>PROJECT SCOPE:</b> Computer Networks Semester Project", table_cell_style),
            Paragraph("<b>TOTAL PAPER MATRIX:</b> 45 Runs (Strict Paper Replication)", table_cell_style)
        ],
        [
            Paragraph("<b>BASE PAPER:</b> Wang et al. (Journal of Supercomputing, 2024)", table_cell_style),
            Paragraph("<b>GITHUB REPOSITORY:</b> <code>Harish8955/IoT_NIDS_Project</code>", table_cell_style)
        ]
    ]
    t_meta = Table(meta_data, colWidths=[260, 280])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), ALERT_BG),
        ('BOX', (0, 0), (-1, -1), 1, SECONDARY_COLOR),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 8))

    # Workload Allocation Overview Table
    summary_data = [
        [Paragraph("Group Member", table_header_style), Paragraph("Assigned Paper Dataset & Model Runs", table_header_style), Paragraph("Assigned Runs", table_header_style)],
        [Paragraph("<b>Harish Srinivas</b>", table_cell_style), Paragraph("UNSW-NB15 Proposed Model & Baseline Runs (50, 100, 200 Epochs)", table_cell_style), Paragraph("<b>12 Runs</b>", table_cell_style)],
        [Paragraph("<b>Edwin Sundaraj</b>", table_cell_style), Paragraph("CSE-CIC-IDS2018 Proposed, 1D-CNN & ResNet Runs + UNSW Baselines", table_cell_style), Paragraph("<b>11 Runs</b>", table_cell_style)],
        [Paragraph("<b>Anish Thondepu</b>", table_cell_style), Paragraph("CIC-IOT2023 Proposed, 1D-CNN & ResNet Runs + UNSW Baselines", table_cell_style), Paragraph("<b>11 Runs</b>", table_cell_style)],
        [Paragraph("<b>Lakshan</b>", table_cell_style), Paragraph("CSE-CIC-IDS2018 & CIC-IOT2023 ResNeSt and ResNet-BiGRU Baselines", table_cell_style), Paragraph("<b>11 Runs</b>", table_cell_style)]
    ]
    t_sum = Table(summary_data, colWidths=[130, 310, 100])
    t_sum.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY_COLOR),
        ('BACKGROUND', (0,1), (-1,-1), BG_LIGHT),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_sum)
    story.append(Spacer(1, 8))

    # =========================================================================
    # HARISH SRINIVAS: 12 RUNS
    # =========================================================================
    story.append(Paragraph("1. HARISH SRINIVAS — ASSIGNED PAPER RUNS (12 RUNS)", h1_style))
    code_h = (
        "python train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model proposed --epochs 50 --balance\n"
        "python train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model proposed --epochs 100 --balance\n"
        "python train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model proposed --epochs 200 --balance\n"
        "python train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model cnn --epochs 50 --balance\n"
        "python train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model cnn --epochs 100 --balance\n"
        "python train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model cnn --epochs 200 --balance\n"
        "python train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model resnet --epochs 50 --balance\n"
        "python train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model resnet --epochs 100 --balance\n"
        "python train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model resnet --epochs 200 --balance\n"
        "python train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model resnest --epochs 200 --balance\n"
        "python train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model resnet-gru --epochs 100 --balance\n"
        "python train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model resnet-gru --epochs 200 --balance"
    )
    t_h = Table([[Paragraph(f"<font face='Courier' size='7.5'>{code_h.replace('\n', '<br/>').replace(' ', '&nbsp;')}</font>", table_cell_style)]], colWidths=[540])
    t_h.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), BG_LIGHT),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_h)
    story.append(Spacer(1, 8))

    # =========================================================================
    # EDWIN SUNDARAJ: 11 RUNS
    # =========================================================================
    story.append(Paragraph("2. EDWIN SUNDARAJ — ASSIGNED PAPER RUNS (11 RUNS)", h1_style))
    code_e = (
        "python train.py --dataset data/CIC_IDS2018_combined.csv --dataset_name CIC-IDS2018 --target_col Label --model proposed --epochs 50 --balance\n"
        "python train.py --dataset data/CIC_IDS2018_combined.csv --dataset_name CIC-IDS2018 --target_col Label --model proposed --epochs 100 --balance\n"
        "python train.py --dataset data/CIC_IDS2018_combined.csv --dataset_name CIC-IDS2018 --target_col Label --model proposed --epochs 200 --balance\n"
        "python train.py --dataset data/CIC_IDS2018_combined.csv --dataset_name CIC-IDS2018 --target_col Label --model cnn --epochs 50 --balance\n"
        "python train.py --dataset data/CIC_IDS2018_combined.csv --dataset_name CIC-IDS2018 --target_col Label --model cnn --epochs 100 --balance\n"
        "python train.py --dataset data/CIC_IDS2018_combined.csv --dataset_name CIC-IDS2018 --target_col Label --model cnn --epochs 200 --balance\n"
        "python train.py --dataset data/CIC_IDS2018_combined.csv --dataset_name CIC-IDS2018 --target_col Label --model resnet --epochs 50 --balance\n"
        "python train.py --dataset data/CIC_IDS2018_combined.csv --dataset_name CIC-IDS2018 --target_col Label --model resnet --epochs 100 --balance\n"
        "python train.py --dataset data/CIC_IDS2018_combined.csv --dataset_name CIC-IDS2018 --target_col Label --model resnet --epochs 200 --balance\n"
        "python train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model resnest --epochs 50 --balance\n"
        "python train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model resnest --epochs 100 --balance"
    )
    t_e = Table([[Paragraph(f"<font face='Courier' size='7.5'>{code_e.replace('\n', '<br/>').replace(' ', '&nbsp;')}</font>", table_cell_style)]], colWidths=[540])
    t_e.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), BG_LIGHT),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_e)
    story.append(Spacer(1, 8))

    # =========================================================================
    # ANISH THONDEPU: 11 RUNS
    # =========================================================================
    story.append(Paragraph("3. ANISH THONDEPU — ASSIGNED PAPER RUNS (11 RUNS)", h1_style))
    code_a = (
        "python train.py --dataset data/CICIOT23_combined.csv --dataset_name CIC-IOT2023 --target_col label --model proposed --epochs 50 --balance\n"
        "python train.py --dataset data/CICIOT23_combined.csv --dataset_name CIC-IOT2023 --target_col label --model proposed --epochs 100 --balance\n"
        "python train.py --dataset data/CICIOT23_combined.csv --dataset_name CIC-IOT2023 --target_col label --model proposed --epochs 200 --balance\n"
        "python train.py --dataset data/CICIOT23_combined.csv --dataset_name CIC-IOT2023 --target_col label --model cnn --epochs 50 --balance\n"
        "python train.py --dataset data/CICIOT23_combined.csv --dataset_name CIC-IOT2023 --target_col label --model cnn --epochs 100 --balance\n"
        "python train.py --dataset data/CICIOT23_combined.csv --dataset_name CIC-IOT2023 --target_col label --model cnn --epochs 200 --balance\n"
        "python train.py --dataset data/CICIOT23_combined.csv --dataset_name CIC-IOT2023 --target_col label --model resnet --epochs 50 --balance\n"
        "python train.py --dataset data/CICIOT23_combined.csv --dataset_name CIC-IOT2023 --target_col label --model resnet --epochs 100 --balance\n"
        "python train.py --dataset data/CICIOT23_combined.csv --dataset_name CIC-IOT2023 --target_col label --model resnet --epochs 200 --balance\n"
        "python train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model resnet-gru --epochs 50 --balance\n"
        "python train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model resnet-gru --epochs 100 --balance"
    )
    t_a = Table([[Paragraph(f"<font face='Courier' size='7.5'>{code_a.replace('\n', '<br/>').replace(' ', '&nbsp;')}</font>", table_cell_style)]], colWidths=[540])
    t_a.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), BG_LIGHT),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_a)
    story.append(Spacer(1, 8))

    # =========================================================================
    # LAKSHAN: 11 RUNS
    # =========================================================================
    story.append(Paragraph("4. LAKSHAN — ASSIGNED PAPER RUNS (11 RUNS)", h1_style))
    code_l = (
        "python train.py --dataset data/CIC_IDS2018_combined.csv --dataset_name CIC-IDS2018 --target_col Label --model resnest --epochs 50 --balance\n"
        "python train.py --dataset data/CIC_IDS2018_combined.csv --dataset_name CIC-IDS2018 --target_col Label --model resnest --epochs 100 --balance\n"
        "python train.py --dataset data/CIC_IDS2018_combined.csv --dataset_name CIC-IDS2018 --target_col Label --model resnest --epochs 200 --balance\n"
        "python train.py --dataset data/CIC_IDS2018_combined.csv --dataset_name CIC-IDS2018 --target_col Label --model resnet-gru --epochs 50 --balance\n"
        "python train.py --dataset data/CIC_IDS2018_combined.csv --dataset_name CIC-IDS2018 --target_col Label --model resnet-gru --epochs 100 --balance\n"
        "python train.py --dataset data/CIC_IDS2018_combined.csv --dataset_name CIC-IDS2018 --target_col Label --model resnet-gru --epochs 200 --balance\n"
        "python train.py --dataset data/CICIOT23_combined.csv --dataset_name CIC-IOT2023 --target_col label --model resnest --epochs 50 --balance\n"
        "python train.py --dataset data/CICIOT23_combined.csv --dataset_name CIC-IOT2023 --target_col label --model resnest --epochs 100 --balance\n"
        "python train.py --dataset data/CICIOT23_combined.csv --dataset_name CIC-IOT2023 --target_col label --model resnest --epochs 200 --balance\n"
        "python train.py --dataset data/CICIOT23_combined.csv --dataset_name CIC-IOT2023 --target_col label --model resnet-gru --epochs 50 --balance\n"
        "python train.py --dataset data/CICIOT23_combined.csv --dataset_name CIC-IOT2023 --target_col label --model resnet-gru --epochs 100 --balance"
    )
    t_l = Table([[Paragraph(f"<font face='Courier' size='7.5'>{code_l.replace('\n', '<br/>').replace(' ', '&nbsp;')}</font>", table_cell_style)]], colWidths=[540])
    t_l.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), BG_LIGHT),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_l)
    story.append(Spacer(1, 8))

    # =========================================================================
    # GIT SYNC BACK WORKFLOW
    # =========================================================================
    story.append(Paragraph("5. Syncing Results Back to Master GitHub Repository", h1_style))
    story.append(Paragraph("When training completes, each group member syncs their generated results back to the master Excel summary table (`results/experiment_summary.csv`):", body_style))
    
    code_sync = (
        "git pull\n"
        "git add results/\n"
        "git commit -m \"Added experiment training results\"\n"
        "git push"
    )
    t_sync = Table([[Paragraph(f"<font face='Courier' size='8'>{code_sync.replace('\n', '<br/>').replace(' ', '&nbsp;')}</font>", table_cell_style)]], colWidths=[540])
    t_sync.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), BG_LIGHT),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_sync)

    doc.build(story)
    print(f"Group Workload Division PDF created successfully at: {filename}")

if __name__ == '__main__':
    build_workload_pdf()
