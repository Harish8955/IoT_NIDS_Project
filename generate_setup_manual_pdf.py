import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

def build_setup_manual_pdf(filename="Teammates_Setup_Manual.pdf"):
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
    story.append(Paragraph("Teammate Setup & Training Manual (Git Collaboration Guide)", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=SECONDARY_COLOR, spaceAfter=8))

    # Meta Info Box
    meta_data = [
        [
            Paragraph("<b>PROJECT SCOPE:</b> Computer Networks Semester Project", table_cell_style),
            Paragraph("<b>GITHUB REPOSITORY URL:</b>", table_cell_style)
        ],
        [
            Paragraph("<b>BASE PAPER:</b> Wang et al. (Journal of Supercomputing, 2024)", table_cell_style),
            Paragraph("<code>https://github.com/Harish8955/IoT_NIDS_Project.git</code>", table_cell_style)
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

    # =========================================================================
    # STEP 1: CLONE REPO
    # =========================================================================
    story.append(Paragraph("STEP 1: Clone Repository to Desired Laptop Destination", h1_style))
    story.append(Paragraph("Open <b>Command Prompt (cmd.exe)</b> on your laptop and navigate to your preferred working folder (Desktop, Downloads, D: drive, etc.):", body_style))
    
    code1 = (
        "cd /d \"C:\\Users\\YOUR_NAME\\Desktop\"\n"
        "git clone https://github.com/Harish8955/IoT_NIDS_Project.git\n"
        "cd IoT_NIDS_Project"
    )
    t_code1 = Table([[Paragraph(f"<font face='Courier' size='8'>{code1.replace('\n', '<br/>').replace(' ', '&nbsp;')}</font>", table_cell_style)]], colWidths=[540])
    t_code1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), BG_LIGHT),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_code1)
    story.append(Spacer(1, 6))

    # =========================================================================
    # STEP 2: CREATE VENV
    # =========================================================================
    story.append(Paragraph("STEP 2: Create & Activate Local Virtual Environment (10 Seconds)", h1_style))
    story.append(Paragraph("Create a fresh local <code>venv</code> bound specifically to your laptop's drive paths:", body_style))
    
    code2 = (
        "python -m venv venv\n"
        "venv\\Scripts\\activate.bat"
    )
    t_code2 = Table([[Paragraph(f"<font face='Courier' size='8'>{code2.replace('\n', '<br/>').replace(' ', '&nbsp;')}</font>", table_cell_style)]], colWidths=[540])
    t_code2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), BG_LIGHT),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_code2)
    story.append(Spacer(1, 6))

    # =========================================================================
    # STEP 3: INSTALL DEPENDENCIES
    # =========================================================================
    story.append(Paragraph("STEP 3: Install Dependencies (Single Command)", h1_style))
    story.append(Paragraph("Install PyTorch, CTGAN, Scikit-Learn, Pandas, ReportLab, and all 13 required libraries:", body_style))
    
    code3 = "pip install -r requirements.txt"
    t_code3 = Table([[Paragraph(f"<font face='Courier' size='8'>{code3}</font>", table_cell_style)]], colWidths=[540])
    t_code3.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), BG_LIGHT),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_code3)
    story.append(Paragraph("<i>Note for NVIDIA GPUs: Install CUDA PyTorch via <code>pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118</code> for 10x faster training.</i>", body_style))
    story.append(Spacer(1, 6))

    # =========================================================================
    # STEP 4: PRE-COMBINED DATASETS (NO COMBINING NEEDED!)
    # =========================================================================
    story.append(Paragraph("STEP 4: Pre-Combined Master Datasets (Ready to Train)", h1_style))
    story.append(Paragraph("The master datasets are <b>already pre-combined</b> inside the shared Google Drive / project package (`data/`). Teammates can skip dataset combining and proceed directly to model training!", body_style))
    story.append(Paragraph("• <code>data/UNSW_NB15_combined.csv</code> (Pre-combined UNSW-NB15 dataset)<br/>"
                           "• <code>data/CIC_IDS2018_combined.csv</code> (Pre-combined CSE-CIC-IDS2018 dataset)<br/>"
                           "• <code>data/CICIOT23_combined.csv</code> (Pre-combined CIC-IOT2023 dataset)", body_style))
    story.append(Spacer(1, 6))

    # =========================================================================
    # STEP 5: MODEL TRAINING COMMANDS
    # =========================================================================
    story.append(Paragraph("STEP 5: Execute Assigned Model Training Commands", h1_style))
    
    cmd_data = [
        [Paragraph("Task Category", table_header_style), Paragraph("Command Prompt Execution String", table_header_style)],
        [
            Paragraph("<b>Proposed Model Peak Runs</b>", table_cell_style),
            Paragraph("<code>python train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model proposed --epochs 200 --balance</code>", table_cell_style)
        ],
        [
            Paragraph("<b>Baseline 1D-CNN Run</b>", table_cell_style),
            Paragraph("<code>python train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model cnn --epochs 50 --balance</code>", table_cell_style)
        ],
        [
            Paragraph("<b>Baseline ResNet Run</b>", table_cell_style),
            Paragraph("<code>python train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model resnet --epochs 50 --balance</code>", table_cell_style)
        ],
        [
            Paragraph("<b>Dual-Engine LOCO Zero-Day</b>", table_cell_style),
            Paragraph("<code>python train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model dual-engine --epochs 50 --balance --hide_class Worms</code>", table_cell_style)
        ]
    ]
    t_cmd = Table(cmd_data, colWidths=[140, 400])
    t_cmd.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY_COLOR),
        ('BACKGROUND', (0,1), (-1,-1), BG_LIGHT),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_cmd)
    story.append(Spacer(1, 6))

    # =========================================================================
    # STEP 6: GIT SYNC BACK
    # =========================================================================
    story.append(Paragraph("STEP 6: Push Results Back to Master GitHub Repository", h1_style))
    story.append(Paragraph("When training completes, sync your generated metrics back to the team's master Excel summary sheet:", body_style))
    
    code6 = (
        "git pull\n"
        "git add results/\n"
        "git commit -m \"Added experiment training results\"\n"
        "git push"
    )
    t_code6 = Table([[Paragraph(f"<font face='Courier' size='8'>{code6.replace('\n', '<br/>').replace(' ', '&nbsp;')}</font>", table_cell_style)]], colWidths=[540])
    t_code6.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), BG_LIGHT),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_code6)

    doc.build(story)
    print(f"Teammates Setup Manual PDF created successfully at: {filename}")

if __name__ == '__main__':
    build_setup_manual_pdf()
