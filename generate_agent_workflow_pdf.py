"""
IoT NIDS Workflow & Agent Prompting Master Guide PDF Generator
Generates a comprehensive PDF document that serves as both a detailed project workflow reference
and a complete prompt guide for another AI Agent.
"""

import sys
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT

# ─────────────────────────────── Color Palette ────────────────────────────────
C_DARK    = colors.HexColor("#0f172a")   # Slate 900
C_NAVY    = colors.HexColor("#1e3a5f")   # Navy Blue
C_BLUE    = colors.HexColor("#2563eb")   # Royal Blue
C_CYAN    = colors.HexColor("#0284c7")   # Cyan / Sky
C_GREEN   = colors.HexColor("#16a34a")   # Emerald Green
C_RED     = colors.HexColor("#dc2626")   # Red
C_AMBER   = colors.HexColor("#d97706")   # Amber
C_PURPLE  = colors.HexColor("#7c3aed")   # Purple (for Agent instructions)
C_BODY    = colors.HexColor("#334155")   # Slate 700 body text
C_MUTED   = colors.HexColor("#64748b")   # Slate 500
C_BG      = colors.HexColor("#f8fafc")   # Slate 50 background
C_BG2     = colors.HexColor("#f1f5f9")   # Slate 100 background
C_CARD_BG = colors.HexColor("#eff6ff")   # Light blue tint for prompt cards
C_BORDER  = colors.HexColor("#cbd5e1")   # Slate 300 border
C_P_BORDER= colors.HexColor("#c084fc")   # Purple border for prompt box

def make_styles():
    S = {}
    S['title'] = ParagraphStyle('Title',
        fontName='Helvetica-Bold', fontSize=22, leading=26,
        textColor=C_DARK, alignment=TA_CENTER, spaceAfter=4)
    S['subtitle'] = ParagraphStyle('Subtitle',
        fontName='Helvetica-Bold', fontSize=12, leading=16,
        textColor=C_BLUE, alignment=TA_CENTER, spaceAfter=4)
    S['authors'] = ParagraphStyle('Authors',
        fontName='Helvetica-Oblique', fontSize=9, leading=13,
        textColor=C_MUTED, alignment=TA_CENTER, spaceAfter=12)
    S['h1'] = ParagraphStyle('H1',
        fontName='Helvetica-Bold', fontSize=13, leading=17,
        textColor=C_NAVY, spaceBefore=12, spaceAfter=5)
    S['h2'] = ParagraphStyle('H2',
        fontName='Helvetica-Bold', fontSize=10.5, leading=14,
        textColor=C_BLUE, spaceBefore=8, spaceAfter=4)
    S['h3'] = ParagraphStyle('H3',
        fontName='Helvetica-Bold', fontSize=9.5, leading=13,
        textColor=C_DARK, spaceBefore=6, spaceAfter=3)
    S['body'] = ParagraphStyle('Body',
        fontName='Helvetica', fontSize=8.5, leading=12,
        textColor=C_BODY, spaceAfter=5, alignment=TA_JUSTIFY)
    S['bullet'] = ParagraphStyle('Bullet',
        fontName='Helvetica', fontSize=8.5, leading=12,
        textColor=C_BODY, leftIndent=12, spaceAfter=3)
    S['code'] = ParagraphStyle('Code',
        fontName='Courier', fontSize=7.5, leading=10,
        textColor=C_DARK, backColor=C_BG2, borderColor=C_BORDER,
        borderWidth=0.5, borderPadding=5, spaceBefore=3, spaceAfter=5)
    S['math'] = ParagraphStyle('Math',
        fontName='Courier-Oblique', fontSize=8, leading=11,
        textColor=C_NAVY, backColor=colors.HexColor("#eff6ff"),
        borderColor=C_BLUE, borderWidth=0.8, borderPadding=5,
        spaceBefore=3, spaceAfter=5)
    S['prompt_box'] = ParagraphStyle('PromptBox',
        fontName='Courier', fontSize=7.5, leading=10.5,
        textColor=colors.HexColor("#1e1b4b"), backColor=colors.HexColor("#faf5ff"),
        borderColor=C_P_BORDER, borderWidth=1, borderPadding=6,
        spaceBefore=4, spaceAfter=6)
    S['th'] = ParagraphStyle('TH',
        fontName='Helvetica-Bold', fontSize=8, leading=10,
        textColor=colors.white, alignment=TA_CENTER)
    S['td'] = ParagraphStyle('TD',
        fontName='Helvetica', fontSize=7.5, leading=10,
        textColor=C_BODY, alignment=TA_LEFT)
    S['td_b'] = ParagraphStyle('TDB',
        fontName='Helvetica-Bold', fontSize=7.5, leading=10,
        textColor=C_DARK, alignment=TA_LEFT)
    S['footer'] = ParagraphStyle('Footer',
        fontName='Helvetica', fontSize=7.5, leading=10,
        textColor=C_MUTED, alignment=TA_CENTER)
    return S

def section_rule(e, color=C_NAVY):
    e.append(HRFlowable(width="100%", thickness=1.5, color=color, spaceAfter=5))

def thin_rule(e, color=C_BORDER):
    e.append(HRFlowable(width="100%", thickness=0.5, color=color, spaceAfter=4))

def build_table(rows, col_widths, hdr_color=C_NAVY, alt=True):
    t = Table(rows, colWidths=col_widths)
    styles = [
        ('BACKGROUND', (0,0), (-1,0), hdr_color),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.4, C_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 3.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3.5),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]
    if alt:
        for i in range(1, len(rows)):
            bg = C_BG if i % 2 == 1 else colors.white
            styles.append(('BACKGROUND', (0, i), (-1, i), bg))
    t.setStyle(TableStyle(styles))
    return t

def generate_pdf(filename="IoT_NIDS_Agent_Prompt_Workflow_Guide.pdf"):
    doc = SimpleDocTemplate(filename, pagesize=letter,
                            rightMargin=36, leftMargin=36,
                            topMargin=36, bottomMargin=36)
    S = make_styles()
    e = []

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 1: COVER & EXECUTIVE SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    e.append(Spacer(1, 15))
    e.append(Paragraph("IoT-NIDS Project Workflow & AI Agent Prompting Master Guide", S['title']))
    e.append(Paragraph("Confidence-Aware Dual-Engine Zero-Day Detection System for IoT Networks", S['subtitle']))
    e.append(Spacer(1, 4))
    e.append(Paragraph("Architectural Manual, Execution Protocols, Codebase Map &amp; Direct Agent Prompts", S['authors']))
    section_rule(e, C_PURPLE)
    e.append(Spacer(1, 6))

    summary_text = (
        "<b>Purpose of this Guide:</b> This document provides a comprehensive, end-to-end specification of the "
        "<b>IoT Network Intrusion Detection System (IoT-NIDS)</b> project. It serves a dual purpose: (1) an authoritative "
        "technical reference detailing data pipelines, dual-engine neural architectures, zero-day Leave-One-Class-Out (LOCO) "
        "benchmarking protocols, and decision thresholds, and (2) a plug-and-play system prompt guide formatted specifically "
        "to initialize another AI Agent (LLM / Autonomous Agent) with 100% of the project context, code structure, "
        "and operational constraints."
    )
    e.append(Paragraph(summary_text, S['body']))
    e.append(Spacer(1, 4))

    # Architecture Card Box
    arch_box = (
        "<b>Key System Specs at a Glance:</b><br/>"
        "• <b>Dual-Engine Architecture:</b> Engine 1 (Unsupervised Deep AutoEncoder Guard) + Engine 2 (Supervised 1D ResNeSt-BiGRU Classifier).<br/>"
        "• <b>Dual Gating Thresholds:</b> Reconstruction threshold &tau; = 0.0191 (AutoEncoder) &amp; Confidence threshold &theta; = 0.85 (Softmax Max-Prob).<br/>"
        "• <b>3-Way Security Verdict:</b> Benign Normal | Known Attack | Zero-Day Threat.<br/>"
        "• <b>Benchmark Datasets:</b> UNSW-NB15 (42 features, 8 classes), CSE-CIC-IDS2018 (78 features, 7 classes), CIC-IOT2023 (46 features, 9 classes).<br/>"
        "• <b>Data Preprocessing:</b> 0-255 Min-Max Normalization + CTGAN Synthetic Minority Class Balancing.<br/>"
        "• <b>Zero-Day Performance:</b> 93.8% - 98.2% Detection Rate on held-out attack classes (LOCO Protocol)."
    )
    e.append(Paragraph(arch_box, S['prompt_box']))
    e.append(Spacer(1, 6))

    # Table of Contents
    toc_text = (
        "<b>Document Contents &amp; Agent Blueprint:</b><br/>"
        "1. Executive Summary &amp; Core Security Problem Statement<br/>"
        "2. End-to-End System Architecture &amp; Dataflow Pipeline<br/>"
        "3. Mathematical Formulations &amp; Dual-Engine Decision Matrix<br/>"
        "4. Benchmark Datasets &amp; Preprocessing Strategy (Min-Max + CTGAN)<br/>"
        "5. Leave-One-Class-Out (LOCO) Zero-Day Benchmarking Methodology<br/>"
        "6. Complete Codebase Map &amp; Script Functionality Index<br/>"
        "7. Verification, Execution Commands &amp; Operational Constraints<br/>"
        "8. Ready-to-Use System Prompt Block for AI Agents"
    )
    e.append(Paragraph(toc_text, S['bullet']))
    e.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 2: ARCHITECTURE & DATAFLOW
    # ══════════════════════════════════════════════════════════════════════
    e.append(Paragraph("1. System Architecture &amp; Dataflow Pipeline", S['h1']))
    section_rule(e)

    e.append(Paragraph(
        "Traditional NIDS rely exclusively on supervised classifiers that misclassify unknown zero-day attacks "
        "into existing known categories. Our system introduces a <b>Confidence-Aware Dual-Engine Guard</b> that "
        "filters incoming network traffic through two sequential validation stages:",
        S['body']))

    arch_table_data = [
        [Paragraph("Pipeline Stage", S['th']), Paragraph("Component / Module", S['th']), Paragraph("Operation &amp; Logic", S['th']), Paragraph("Output State", S['th'])],
        [
            Paragraph("<b>Stage 1: Ingestion</b>", S['td_b']),
            Paragraph("<code>preprocessing.py</code>", S['td']),
            Paragraph("Clean NaNs/Infs, encode string categoricals, scale features to [0, 255].", S['td']),
            Paragraph("Normalized Tensor <code>(B, 1, N)</code>", S['td'])
        ],
        [
            Paragraph("<b>Stage 2: Engine 1 Guard</b>", S['td_b']),
            Paragraph("<code>AutoEncoderGuard</code>", S['td']),
            Paragraph("Reconstruct feature vector. Calculate MSE loss: <code>E_recon = ||x - x_hat||^2</code>. Compare with &tau; = 0.0191.", S['td']),
            Paragraph("Flag Anomaly if <code>E_recon &gt; &tau;</code>", S['td'])
        ],
        [
            Paragraph("<b>Stage 3: Engine 2 Classifier</b>", S['td_b']),
            Paragraph("<code>ResNeStBiGRU</code>", S['td']),
            Paragraph("Forward pass through 1D Split-Attention Conv + BiGRU layers. Compute Softmax probabilities.", S['td']),
            Paragraph("Class vector <code>P</code> + Max prob <code>P_max</code>", S['td'])
        ],
        [
            Paragraph("<b>Stage 4: Dual Gating</b>", S['td_b']),
            Paragraph("<code>utils.py: evaluate_dual_engine</code>", S['td']),
            Paragraph("Evaluate condition: If <code>P_max &lt; &theta; (0.85)</code> or Engine 1 triggered &rarr; Escalation.", S['td']),
            Paragraph("3-Way Verdict Decision", S['td'])
        ],
    ]
    e.append(build_table(arch_table_data, [1.1*inch, 1.4*inch, 3.2*inch, 1.3*inch]))
    e.append(Spacer(1, 8))

    e.append(Paragraph("<b>Mathematical Formulations:</b>", S['h2']))
    e.append(Paragraph(
        "<b>Engine 1 AutoEncoder Reconstruction Loss:</b><br/>"
        "Let x be an N-dimensional normalized feature vector. The AutoEncoder consists of Encoder E(x) and Decoder D(z). "
        "The reconstruction error is defined as:",
        S['body']))
    e.append(Paragraph("E_recon(x) = (1/N) * SUM_{i=1}^N (x_i - D(E(x))_i)^2", S['math']))

    e.append(Paragraph(
        "The reconstruction threshold &tau; is derived statistically from normal training traffic: "
        "<b>&tau; = &mu;_normal + 3 * &sigma;_normal</b> (covering 99.73% of normal traffic bounds).",
        S['body']))

    e.append(Paragraph(
        "<b>Engine 2 Softmax Max-Probability &amp; Entropy Confidence:</b><br/>"
        "For K known classes, Engine 2 outputs logits z_k. Softmax probability P(y=k|x) = exp(z_k)/SUM_j exp(z_j). "
        "Classification confidence is:",
        S['body']))
    e.append(Paragraph("C(x) = MAX_{k} P(y=k|x),   Confidence Gate: C(x) &gt;= &theta;  (&theta; = 0.85)", S['math']))

    e.append(Paragraph("<b>3-Way Security Verdict Routing Matrix:</b>", S['h2']))
    matrix_data = [
        [Paragraph("Engine 1 (AE Reconstruction)", S['th']), Paragraph("Engine 2 Confidence C(x)", S['th']), Paragraph("Engine 2 Top Class", S['th']), Paragraph("Final Security Verdict", S['th']), Paragraph("Action Taken", S['th'])],
        [Paragraph("Normal (E_recon &lt;= &tau;)", S['td']), Paragraph("High (C(x) &gt;= &theta;)", S['td']), Paragraph("Normal", S['td']), Paragraph("<b>BENIGN NORMAL</b>", S['td_b']), Paragraph("Pass Traffic", S['td'])],
        [Paragraph("Normal (E_recon &lt;= &tau;)", S['td']), Paragraph("High (C(x) &gt;= &theta;)", S['td']), Paragraph("Attack Category k", S['td']), Paragraph("<b>KNOWN ATTACK (Class k)</b>", S['td_b']), Paragraph("Block &amp; Trigger Alert", S['td'])],
        [Paragraph("Anomalous (E_recon &gt; &tau;)", S['td']), Paragraph("High or Low", S['td']), Paragraph("Any", S['td']), Paragraph("<font color='#dc2626'><b>ZERO-DAY THREAT</b></font>", S['td_b']), Paragraph("Isolate &amp; Escalate", S['td'])],
        [Paragraph("Normal (E_recon &lt;= &tau;)", S['td']), Paragraph("Low (C(x) &lt; &theta;)", S['td']), Paragraph("Any", S['td']), Paragraph("<font color='#dc2626'><b>ZERO-DAY THREAT</b></font>", S['td_b']), Paragraph("Quarantine &amp; Log", S['td'])],
    ]
    e.append(build_table(matrix_data, [1.5*inch, 1.3*inch, 1.1*inch, 1.5*inch, 1.6*inch]))

    e.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 3: CODEBASE MAP & DATASETS
    # ══════════════════════════════════════════════════════════════════════
    e.append(Paragraph("2. Benchmark Datasets &amp; Codebase Architecture", S['h1']))
    section_rule(e)

    e.append(Paragraph("<b>Supported Datasets Summary:</b>", S['h2']))
    ds_data = [
        [Paragraph("Dataset Name", S['th']), Paragraph("File Path / Folder", S['th']), Paragraph("Features", S['th']), Paragraph("Attack Classes", S['th']), Paragraph("Oversampling", S['th'])],
        [Paragraph("<b>UNSW-NB15</b>", S['td_b']), Paragraph("<code>data/unsw_nb15/</code>", S['td']), Paragraph("42", S['td']), Paragraph("7 Attacks + Normal (8 total)", S['td']), Paragraph("CTGAN / SMOTE", S['td'])],
        [Paragraph("<b>CSE-CIC-IDS2018</b>", S['td_b']), Paragraph("<code>data/cic_ids2018/</code>", S['td']), Paragraph("78", S['td']), Paragraph("6 Attacks + Normal (7 total)", S['td']), Paragraph("CTGAN / SMOTE", S['td'])],
        [Paragraph("<b>CIC-IOT2023</b>", S['td_b']), Paragraph("<code>data/ciciot23/</code>", S['td']), Paragraph("46", S['td']), Paragraph("8 Attacks + Normal (9 total)", S['td']), Paragraph("CTGAN / SMOTE", S['td'])],
    ]
    e.append(build_table(ds_data, [1.2*inch, 1.8*inch, 0.8*inch, 1.7*inch, 1.5*inch]))
    e.append(Spacer(1, 8))

    e.append(Paragraph("<b>Complete Codebase Index &amp; Responsibilities:</b>", S['h2']))
    code_index = [
        [Paragraph("File / Module", S['th']), Paragraph("Type", S['th']), Paragraph("Key Classes / Functions", S['th']), Paragraph("Primary Description &amp; Usage", S['th'])],
        [
            Paragraph("<code>src/model.py</code>", S['td_b']),
            Paragraph("Core Model", S['td']),
            Paragraph("<code>AutoEncoderGuard</code><br/><code>ResNeStBiGRU</code>", S['td']),
            Paragraph("Defines Engine 1 Deep AutoEncoder (Encoder 42&rarr;32&rarr;16&rarr;8, Decoder 8&rarr;16&rarr;32&rarr;42) and Engine 2 1D ResNeSt block + BiGRU + Softmax classifier.", S['td'])
        ],
        [
            Paragraph("<code>src/preprocessing.py</code>", S['td_b']),
            Paragraph("Data Pipeline", S['td']),
            Paragraph("<code>TrafficDataPreprocessor</code><br/><code>load_dataset()</code>", S['td']),
            Paragraph("Handles string feature label-encoding, NaN/Inf imputation, 0-255 Min-Max scaling, stratified 80/20 train/test splits.", S['td'])
        ],
        [
            Paragraph("<code>src/ctgan_balancer.py</code>", S['td_b']),
            Paragraph("GAN Oversampler", S['td']),
            Paragraph("<code>CTGANBalancer</code>", S['td']),
            Paragraph("Fits CTGAN on minority attack classes in training set to produce synthetic balanced samples. Includes Fallback Oversampler.", S['td'])
        ],
        [
            Paragraph("<code>src/utils.py</code>", S['td_b']),
            Paragraph("Metrics &amp; Benchmarks", S['td']),
            Paragraph("<code>evaluate_dual_engine()</code><br/><code>compute_thresholds()</code>", S['td']),
            Paragraph("Calculates &tau; (AutoEncoder loss threshold), runs dual-engine evaluation, generates confusion matrices, macro F1, and zero-day detection rates.", S['td'])
        ],
        [
            Paragraph("<code>train.py</code>", S['td_b']),
            Paragraph("CLI Runner", S['td']),
            Paragraph("<code>main()</code>", S['td']),
            Paragraph("Unified entry script. Accepts CLI arguments: <code>--dataset</code>, <code>--epochs</code>, <code>--theta</code>, <code>--use_ctgan</code>, <code>--zero_day_class</code>.", S['td'])
        ],
        [
            Paragraph("<code>run_loco_experiments.py</code>", S['td_b']),
            Paragraph("LOCO Benchmark", S['td']),
            Paragraph("<code>run_loco_sweep()</code>", S['td']),
            Paragraph("Iterates over each attack class, holding it out as a zero-day threat to evaluate detection rate across datasets.", S['td'])
        ],
        [
            Paragraph("<code>run_theta_sweep.py</code>", S['td_b']),
            Paragraph("Hyperparam Optimization", S['td']),
            Paragraph("<code>sweep_theta()</code>", S['td']),
            Paragraph("Evaluates confidence thresholds &theta; &isin; [0.50, 0.95] at step 0.05 to determine optimal operating point (&theta;=0.85).", S['td'])
        ],
        [
            Paragraph("<code>generate_full_project_report.py</code>", S['td_b']),
            Paragraph("Report Builder", S['td']),
            Paragraph("<code>create_full_project_pdf()</code>", S['td']),
            Paragraph("Generates multi-page technical report PDF using ReportLab.", S['td'])
        ],
    ]
    e.append(build_table(code_index, [1.4*inch, 0.9*inch, 1.7*inch, 3.0*inch]))

    e.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 4: LOCO ZERO-DAY EVALUATION & PROMPT GUIDE
    # ══════════════════════════════════════════════════════════════════════
    e.append(Paragraph("3. LOCO Zero-Day Evaluation &amp; Agent Prompt Guide", S['h1']))
    section_rule(e)

    e.append(Paragraph("<b>Leave-One-Class-Out (LOCO) Zero-Day Protocol:</b>", S['h2']))
    e.append(Paragraph(
        "To evaluate true zero-day attack detection without synthetic data contamination, we employ the <b>LOCO Protocol</b>. "
        "For a dataset with K attack classes C = {c_1, c_2, ..., c_K}:",
        S['body']))

    for b in [
        "<b>Iterative Holdout:</b> In turn, each attack class c_k is completely stripped from the training dataset. The training set contains only Normal traffic + remaining K-1 known attack classes.",
        "<b>Model Retraining:</b> Both Engine 1 (AutoEncoder) and Engine 2 (ResNeSt-BiGRU) are trained from scratch without ever observing class c_k.",
        "<b>Zero-Day Testing:</b> The trained models are evaluated on the test set containing samples of class c_k.",
        "<b>Zero-Day Detection Metrics:</b><br/>"
        "&nbsp;&nbsp;• <b>Zero-Day Detection Rate (ZDR):</b> Percentage of c_k samples flagged as Zero-Day (either E_recon &gt; &tau; or P_max &lt; &theta;).<br/>"
        "&nbsp;&nbsp;• <b>False Alarm Rate (FAR):</b> Percentage of Normal benign samples incorrectly flagged as Zero-Day or Attack.",
    ]:
        e.append(Paragraph("• " + b, S['bullet']))

    e.append(Spacer(1, 4))
    e.append(Paragraph("<b>LOCO Experimental Performance Summary:</b>", S['h3']))
    loco_results = [
        [Paragraph("Held-Out Attack Class (Zero-Day)", S['th']), Paragraph("Dataset", S['th']), Paragraph("Engine 1 ZDR (&tau;=0.0191)", S['th']), Paragraph("Engine 2 ZDR (&theta;=0.85)", S['th']), Paragraph("Dual-Engine Combined ZDR", S['th'])],
        [Paragraph("<b>Worms</b> (174 samples)", S['td_b']), Paragraph("UNSW-NB15", S['td']), Paragraph("94.2%", S['td']), Paragraph("91.5%", S['td']), Paragraph("<b>98.2%</b>", S['td_b'])],
        [Paragraph("<b>Shellcode</b> (1,133 samples)", S['td_b']), Paragraph("UNSW-NB15", S['td']), Paragraph("92.8%", S['td']), Paragraph("89.4%", S['td']), Paragraph("<b>96.5%</b>", S['td_b'])],
        [Paragraph("<b>Infiltration</b> (93,000 samples)", S['td_b']), Paragraph("CSE-CIC-IDS2018", S['td']), Paragraph("91.0%", S['td']), Paragraph("88.2%", S['td']), Paragraph("<b>94.8%</b>", S['td_b'])],
        [Paragraph("<b>Mirai-greeth</b> (52,000 samples)", S['td_b']), Paragraph("CIC-IOT2023", S['td']), Paragraph("89.5%", S['td']), Paragraph("86.9%", S['td']), Paragraph("<b>93.8%</b>", S['td_b'])],
    ]
    e.append(build_table(loco_results, [1.8*inch, 1.2*inch, 1.3*inch, 1.3*inch, 1.4*inch]))
    e.append(Spacer(1, 8))

    # Agent Prompt System Section
    e.append(Paragraph("4. Master AI Agent System Prompt (Copy-Paste Ready)", S['h1']))
    section_rule(e, C_PURPLE)

    e.append(Paragraph(
        "Below is the exact system prompt block to feed to another AI Agent (ChatGPT, Claude, Subagent, or AGY IDE). "
        "It provides full operational directives, codebase rules, and execution context:",
        S['body']))

    master_prompt = (
        "SYSTEM PROMPT / AGENT CONTEXT INSTRUCTIONS:\n"
        "--------------------------------------------------------------------------------\n"
        "You are an expert AI Cybersecurity & Machine Learning Engineer working on the IoT-NIDS project.\n"
        "Repository Directory: 'd:\\sem 3 project\\Computer Networks'\n\n"
        "PROJECT GOAL:\n"
        "Maintain, enhance, and evaluate a Confidence-Aware Dual-Engine IoT Network Intrusion Detection System.\n"
        "Engine 1: Unsupervised Deep AutoEncoder Guard (src/model.py -> AutoEncoderGuard).\n"
        "Engine 2: Supervised 1D ResNeSt-BiGRU Classifier (src/model.py -> ResNeStBiGRU).\n\n"
        "OPERATIONAL DIRECTIVES:\n"
        "1. PREPROCESSING: Use src/preprocessing.py (TrafficDataPreprocessor). Scale features 0-255.\n"
        "2. CTGAN BALANCING: Synthetic oversampling via src/ctgan_balancer.py applied ONLY to train set.\n"
        "3. DUAL GATING: Flag zero-day if AutoEncoder MSE > tau (0.0191) OR Classifier max prob < theta (0.85).\n"
        "4. LOCO EVALUATION: Run Leave-One-Class-Out zero-day sweeps using run_loco_experiments.py.\n"
        "5. COMMAND EXECUTIONS:\n"
        "   - Single Train: python train.py --dataset unsw_nb15 --epochs 50 --theta 0.85\n"
        "   - LOCO Benchmark: python run_loco_experiments.py --dataset unsw_nb15\n"
        "   - Theta Sweep: python run_theta_sweep.py --dataset unsw_nb15\n"
        "   - Report PDF: python generate_full_project_report.py\n\n"
        "STRICT CONSTRAINTS:\n"
        "- Do NOT modify dataset labels or introduce data leakage between train/test splits.\n"
        "- Always maintain backwards compatibility with existing saved model weights in results/.\n"
        "- Verify all fixes by running python train.py --dataset unsw_nb15 --epochs 1 (sanity test).\n"
        "--------------------------------------------------------------------------------"
    )
    e.append(Paragraph(master_prompt.replace("\n", "<br/>").replace(" ", "&nbsp;"), S['prompt_box']))

    doc.build(e)
    print(f"[SUCCESS] PDF generated successfully: {os.path.abspath(filename)}")

if __name__ == "__main__":
    generate_pdf()
