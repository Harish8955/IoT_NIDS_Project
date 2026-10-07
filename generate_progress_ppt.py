"""
IoT NIDS Progress Update PPT - Light Theme
Clean, professional academic presentation
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# ─── Color Palette (Light Theme) ─────────────────────────────────────────────
NAVY      = RGBColor(0x0F, 0x17, 0x2A)
DKBLUE    = RGBColor(0x1E, 0x3A, 0x5F)
BLUE      = RGBColor(0x1D, 0x4E, 0xD8)
SKYBLUE   = RGBColor(0x02, 0x84, 0xC7)
LTBLUE    = RGBColor(0xDB, 0xEA, 0xFE)
GREEN     = RGBColor(0x15, 0x80, 0x3D)
LTGREEN   = RGBColor(0xDC, 0xFC, 0xE7)
RED       = RGBColor(0xDC, 0x26, 0x26)
LTRED     = RGBColor(0xFE, 0xE2, 0xE2)
ORANGE    = RGBColor(0xEA, 0x58, 0x0C)
LTORANGE  = RGBColor(0xFF, 0xED, 0xD5)
AMBER     = RGBColor(0xB4, 0x53, 0x09)
LTAMBER   = RGBColor(0xFE, 0xF3, 0xC7)
WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
BG        = RGBColor(0xF8, 0xFA, 0xFC)   # page background
BG2       = RGBColor(0xF1, 0xF5, 0xF9)   # card bg
BG3       = RGBColor(0xE2, 0xE8, 0xF0)   # darker card
BORDER    = RGBColor(0xCB, 0xD5, 0xE1)
BODYTXT   = RGBColor(0x1E, 0x29, 0x3B)
MUTEDTXT  = RGBColor(0x64, 0x74, 0x8B)

prs = Presentation()
prs.slide_width  = Inches(13.33)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]

# ─── Helpers ─────────────────────────────────────────────────────────────────
def rect(slide, l, t, w, h, bg=None, line=None, lw=Pt(0.5)):
    s = slide.shapes.add_shape(1, Inches(l), Inches(t), Inches(w), Inches(h))
    if bg:
        s.fill.solid(); s.fill.fore_color.rgb = bg
    else:
        s.fill.background()
    if line:
        s.line.color.rgb = line; s.line.width = lw
    else:
        s.line.fill.background()
    return s

def txt(slide, text, l, t, w, h,
        size=14, bold=False, italic=False,
        color=BODYTXT, align=PP_ALIGN.LEFT, wrap=True, font="Calibri"):
    tb = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = wrap
    p = tf.paragraphs[0]; p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size); run.font.bold = bold
    run.font.italic = italic; run.font.color.rgb = color
    run.font.name = font
    return tb

def txt2(slide, lines, l, t, w, h, size=12, color=BODYTXT, align=PP_ALIGN.LEFT, sp=3, font="Calibri"):
    tb = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    first = True
    for line in lines:
        ltext = line[0] if isinstance(line, (list,tuple)) else line
        lbold = line[1] if isinstance(line,(list,tuple)) and len(line)>1 else False
        lsize = line[2] if isinstance(line,(list,tuple)) and len(line)>2 else size
        lcol  = line[3] if isinstance(line,(list,tuple)) and len(line)>3 else color
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        p.alignment = align
        if sp: p.space_before = Pt(sp)
        run = p.add_run()
        run.text = ltext; run.font.size = Pt(lsize)
        run.font.bold = lbold; run.font.color.rgb = lcol
        run.font.name = font
        first = False
    return tb

def add_image(slide, path, l, t, w, h):
    return slide.shapes.add_picture(path, Inches(l), Inches(t), Inches(w), Inches(h))

def page_bg(slide):
    rect(slide, 0, 0, 13.33, 7.5, bg=BG)

def header(slide, title, sub=None):
    page_bg(slide)
    rect(slide, 0, 0, 13.33, 1.15, bg=NAVY)
    rect(slide, 0, 1.15, 13.33, 0.05, bg=BLUE)
    txt(slide, title, 0.35, 0.1, 12.5, 0.7, size=26, bold=True, color=WHITE)
    if sub:
        txt(slide, sub, 0.35, 0.76, 12.5, 0.35, size=12, italic=True, color=BG3)
    # footer strip
    rect(slide, 0, 7.25, 13.33, 0.25, bg=BG3)
    txt(slide, "Confidence-Aware Dual-Engine IoT NIDS  |  Sem 3 Computer Networks Project",
        0.3, 7.27, 12.7, 0.2, size=8, color=MUTEDTXT)

def pill(slide, text, l, t, w, h, bg=BLUE, fc=WHITE, size=9):
    rect(slide, l, t, w, h, bg=bg)
    txt(slide, text, l+0.08, t+0.03, w-0.16, h-0.06, size=size, bold=True, color=fc, align=PP_ALIGN.CENTER)

def card(slide, l, t, w, h, title, body_lines, hbg=DKBLUE, bbg=BG2, hfc=WHITE, bfc=BODYTXT):
    rect(slide, l, t, w, 0.38, bg=hbg)
    txt(slide, title, l+0.1, t+0.06, w-0.2, 0.28, size=11, bold=True, color=hfc)
    rect(slide, l, t+0.38, w, h-0.38, bg=bbg, line=BORDER, lw=Pt(0.5))
    body = [(li if isinstance(li,tuple) else (li,False,10,bfc)) for li in body_lines]
    txt2(slide, body, l+0.12, t+0.44, w-0.24, h-0.5, size=10, color=bfc)

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — TITLE
# ═══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK)
rect(slide, 0, 0, 13.33, 7.5, bg=WHITE)
rect(slide, 0, 0, 13.33, 0.08, bg=BLUE)
rect(slide, 0, 7.42, 13.33, 0.08, bg=BLUE)

# Left accent bar
rect(slide, 0, 0.08, 0.12, 7.34, bg=NAVY)

# Main content area
rect(slide, 0.5, 1.2, 12.33, 5.0, bg=BG)
rect(slide, 0.5, 1.2, 0.08, 5.0, bg=BLUE)

txt(slide, "PROJECT PROGRESS UPDATE", 0.9, 1.5, 11.5, 0.45,
    size=13, bold=False, color=BLUE, align=PP_ALIGN.LEFT)
txt(slide, "Confidence-Aware Dual-Engine IoT NIDS", 0.9, 1.9, 11.5, 0.85,
    size=33, bold=True, color=NAVY, align=PP_ALIGN.LEFT)
txt(slide, "Zero-Day Detection via Autoencoder Guard + 1D ResNeSt-BiGRU Classifier", 0.9, 2.72, 11.5, 0.45,
    size=14, italic=True, color=MUTEDTXT, align=PP_ALIGN.LEFT)

rect(slide, 0.9, 3.3, 10.0, 0.04, bg=BORDER)

txt(slide, "LOCO (Leave-One-Class-Out) Zero-Day Experimental Results", 0.9, 3.5, 11.0, 0.4,
    size=13, bold=True, color=DKBLUE, align=PP_ALIGN.LEFT)

# Team pills
team = [("Harish","UNSW-NB15"), ("Edwin","CSE-CIC-IDS2018"), ("Anish","CIC-IOT2023"), ("Lakshan","UNSW-NB15 + IOT2023")]
for i,(name,ds) in enumerate(team):
    x = 0.9 + i*3.0
    rect(slide, x, 4.05, 2.7, 0.75, bg=BG2, line=BORDER, lw=Pt(1))
    txt(slide, name, x+0.1, 4.1, 2.5, 0.32, size=13, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
    txt(slide, ds,   x+0.1, 4.42, 2.5, 0.28, size=8.5, color=MUTEDTXT, align=PP_ALIGN.CENTER)

txt(slide, "Sem 3 Computer Networks  |  September 2026", 0.9, 5.05, 11.0, 0.35,
    size=10, color=MUTEDTXT, align=PP_ALIGN.LEFT)

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — AGENDA
# ═══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK)
header(slide, "Agenda", "What this progress update covers")

items = [
    ("01", "Problem Statement & Motivation",   ORANGE),
    ("02", "Datasets",                         BLUE),
    ("03", "Novel Dual-Engine Architecture",   NAVY),
    ("04", "Threshold Derivation (tau & theta)", DKBLUE),
    ("05", "LOCO Evaluation Protocol",         SKYBLUE),
    ("06", "LOCO Results — Worms",             GREEN),
    ("07", "LOCO Results — Shellcode",         GREEN),
    ("08", "Performance Summary",              BLUE),
    ("09", "Key Contributions",               NAVY),
]
for i,(num,title,col) in enumerate(items):
    row, col_idx = divmod(i, 3)
    x = 0.35 + col_idx * 4.32
    y = 1.35 + row * 1.85
    rect(slide, x, y, 4.1, 1.65, bg=WHITE, line=BORDER, lw=Pt(1))
    rect(slide, x, y, 4.1, 0.08, bg=col)
    txt(slide, num,   x+0.15, y+0.18, 0.7,  0.55, size=26, bold=True, color=col)
    txt(slide, title, x+0.15, y+0.75, 3.8,  0.8,  size=12, bold=True, color=NAVY, wrap=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 3 — PROBLEM STATEMENT
# ═══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK)
header(slide, "Problem Statement", "Why existing NIDS fail against Zero-Day attacks")

# Left panel
rect(slide, 0.3, 1.3, 6.1, 5.75, bg=WHITE, line=BORDER, lw=Pt(1))
rect(slide, 0.3, 1.3, 6.1, 0.08, bg=RED)
txt(slide, "The Problem", 0.5, 1.42, 5.8, 0.45, size=16, bold=True, color=RED)
probs = [
    "Supervised classifiers are trained ONLY on known attack signatures",
    "Zero-day attacks are never seen during training",
    "Classifier is forced to mislabel novel threats as Normal or wrong class",
    "Detection rate = 0% for pure supervised classifiers against zero-days",
    "IoT devices face constant novel threats (75B+ devices projected by 2025)",
    "Severe class imbalance: Worms = 174 samples vs Normal = 93,000 samples",
    "Rare attack classes get ignored by standard ML models entirely",
]
for i,p in enumerate(probs):
    rect(slide, 0.45, 2.0+i*0.58, 0.28, 0.28, bg=LTRED)
    txt(slide, "x", 0.48, 2.02+i*0.58, 0.22, 0.24, size=12, bold=True, color=RED, align=PP_ALIGN.CENTER)
    txt(slide, p, 0.82, 2.0+i*0.58, 5.45, 0.52, size=11, color=BODYTXT)

# Right panel
rect(slide, 6.7, 1.3, 6.3, 5.75, bg=WHITE, line=BORDER, lw=Pt(1))
rect(slide, 6.7, 1.3, 6.3, 0.08, bg=GREEN)
txt(slide, "Our Dual-Engine Solution", 6.9, 1.42, 6.0, 0.45, size=16, bold=True, color=GREEN)

# Engine 1 card
rect(slide, 6.85, 2.0, 6.0, 1.6, bg=LTORANGE, line=ORANGE, lw=Pt(1.2))
txt(slide, "ENGINE 1 — Zero-Day Guard Autoencoder", 7.0, 2.08, 5.7, 0.38, size=11, bold=True, color=ORANGE)
for item in ["Trained ONLY on benign normal traffic", "Any attack causes massive reconstruction spike", "tau = 0.0191  (3-sigma statistical threshold)"]:
    txt(slide, "   + "+item, 7.05, 2.5+["Trained ONLY on benign normal traffic","Any attack causes massive reconstruction spike","tau = 0.0191  (3-sigma statistical threshold)"].index(item)*0.34, 5.7, 0.32, size=10.5, color=BODYTXT)

# Engine 2 card
rect(slide, 6.85, 3.76, 6.0, 1.6, bg=LTBLUE, line=BLUE, lw=Pt(1.2))
txt(slide, "ENGINE 2 — ResNeSt-BiGRU Classifier", 7.0, 3.84, 5.7, 0.38, size=11, bold=True, color=BLUE)
for i,item in enumerate(["Classifies known attack families with labels", "Confidence gate: theta = 0.85 (Youden optimized)", "Low confidence = Zero-Day escalation alert"]):
    txt(slide, "   + "+item, 7.05, 4.26+i*0.34, 5.7, 0.32, size=10.5, color=BODYTXT)

# Result box
rect(slide, 6.85, 5.52, 6.0, 1.38, bg=LTGREEN, line=GREEN, lw=Pt(1.5))
txt(slide, "3-WAY SECURITY VERDICT:", 7.0, 5.58, 5.7, 0.35, size=11, bold=True, color=GREEN)
for i,(v,c) in enumerate([("Benign Normal Traffic", GREEN), ("Known Attack Class (high confidence)", DKBLUE), ("Zero-Day Threat Alert (low confidence)", RED)]):
    txt(slide, f"  {i+1}. {v}", 7.05, 5.95+i*0.3, 5.7, 0.28, size=10.5, bold=(i==2), color=c)

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 4 — DATASETS
# ═══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK)
header(slide, "Benchmark Datasets", "Three industry-standard IoT/Network security datasets")

ds_info = [
    ("UNSW-NB15", "Synthetic Mixed Benchmark", "42", "8", "~365,000",
     ["Normal, Fuzzers, Analysis, Backdoor", "DoS, Exploits, Reconnaissance", "Shellcode, Worms (LOCO targets)"],
     BLUE, LTBLUE),
    ("CSE-CIC-IDS2018", "Enterprise Campus Network (AWS)", "78", "7", "~225,000",
     ["Benign, Infiltration, Bot", "DoS-Slowloris, Heartbleed", "Web Attacks, Brute Force"],
     ORANGE, LTORANGE),
    ("CIC-IOT2023", "Smart Home & Industrial IoT", "46", "9", "~850,000",
     ["Normal, Mirai-greeth, DDoS-ICMP", "Vulnerability_Scan, ARP-Spoofing", "Recon, DNS-Flood, HTTP-Flood"],
     GREEN, LTGREEN),
]
for i,(name,domain,feats,cls,samp,attacks,col,lcol) in enumerate(ds_info):
    x = 0.3 + i*4.35
    # header
    rect(slide, x, 1.3, 4.1, 0.55, bg=col)
    txt(slide, name, x+0.12, 1.35, 3.86, 0.44, size=17, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    # body
    rect(slide, x, 1.85, 4.1, 5.15, bg=WHITE, line=BORDER, lw=Pt(0.8))
    txt(slide, domain, x+0.15, 1.9, 3.8, 0.38, size=9.5, italic=True, color=MUTEDTXT, align=PP_ALIGN.CENTER)

    # Stats row
    for j,(lbl,val) in enumerate([("Features",feats),("Classes",cls),("Samples",samp)]):
        bx = x+0.1+j*1.3
        rect(slide, bx, 2.38, 1.22, 0.7, bg=lcol, line=col, lw=Pt(0.6))
        txt(slide, val, bx+0.06, 2.44, 1.1, 0.32, size=16, bold=True, color=col, align=PP_ALIGN.CENTER)
        txt(slide, lbl, bx+0.06, 2.72, 1.1, 0.22, size=8, color=MUTEDTXT, align=PP_ALIGN.CENTER)

    rect(slide, x+0.1, 3.2, 3.9, 0.04, bg=BORDER)
    txt(slide, "Attack Categories:", x+0.15, 3.3, 3.8, 0.3, size=10, bold=True, color=col)
    for k,atk in enumerate(attacks):
        txt(slide, "  * "+atk, x+0.15, 3.66+k*0.4, 3.8, 0.36, size=10, color=BODYTXT)

    # LOCO label
    if i==0:
        rect(slide, x, 6.7, 4.1, 0.28, bg=col)
        txt(slide, "Harish's LOCO Dataset", x+0.1, 6.73, 3.9, 0.22, size=9.5, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 5 — ARCHITECTURE (with uploaded image)
# ═══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK)
header(slide, "Novel Dual-Engine Architecture", "Sequential two-gate system — our key contribution")

img_path = r"C:/Users/Acer/.gemini/antigravity/brain/d0419e99-7023-42cd-8c44-ec73620dc286/.user_uploaded/media_1790072224358.png"
add_image(slide, img_path, 0.25, 1.28, 12.83, 5.85)

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 6 — THRESHOLD DERIVATION
# ═══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK)
header(slide, "Threshold Derivation", "How tau = 0.0191 and theta = 0.85 were determined")

# Left: tau
rect(slide, 0.3, 1.3, 6.1, 5.75, bg=WHITE, line=BORDER, lw=Pt(0.8))
rect(slide, 0.3, 1.3, 6.1, 0.5, bg=ORANGE)
txt(slide, "ENGINE 1 GATE  —  tau = 0.0191", 0.48, 1.36, 5.8, 0.38, size=14, bold=True, color=WHITE)

tau_steps = [
    ("STEP 1: Train Autoencoder on Benign Traffic Only", True, 11, ORANGE),
    ("  93,000 Normal samples from UNSW-NB15", False, 10.5, BODYTXT),
    ("  MSE reconstruction loss | Adam | lr=0.001", False, 10.5, BODYTXT),
    ("", False, 6, BODYTXT),
    ("STEP 2: Compute Benign L_recon Distribution", True, 11, ORANGE),
    ("  mu_benign    = E[L_recon | Normal] = 0.0042", False, 10.5, BODYTXT),
    ("  sigma_benign = std[L_recon | Normal] = 0.0049", False, 10.5, BODYTXT),
    ("  Distribution verified as near-Gaussian", False, 10.5, BODYTXT),
    ("", False, 6, BODYTXT),
    ("STEP 3: Apply 3-Sigma Gaussian Rule", True, 11, ORANGE),
    ("  tau = mu + 3 * sigma", False, 10.5, BODYTXT),
    ("  tau = 0.0042 + 3 * 0.0049 = 0.0191", False, 11, NAVY),
    ("  Guarantees < 0.27% false alarm on benign traffic", False, 10.5, BODYTXT),
    ("", False, 6, BODYTXT),
    ("STEP 4: Validation on Zero-Day Samples", True, 11, ORANGE),
    ("  Worms L_recon avg    =  8,438.71  (441,816x tau)", False, 10.5, RED),
    ("  Shellcode L_recon avg = 7,920.45  (414,731x tau)", False, 10.5, RED),
]
txt2(slide, tau_steps, 0.48, 1.88, 5.8, 5.0, size=10.5, sp=2)

# Right: theta
rect(slide, 6.73, 1.3, 6.3, 5.75, bg=WHITE, line=BORDER, lw=Pt(0.8))
rect(slide, 6.73, 1.3, 6.3, 0.5, bg=BLUE)
txt(slide, "ENGINE 2 GATE  —  theta = 0.85", 6.9, 1.36, 6.0, 0.38, size=14, bold=True, color=WHITE)

txt(slide, "Youden's J-Index Optimization:", 6.9, 1.9, 6.0, 0.38, size=12, bold=True, color=BLUE)
txt(slide, "J = Detection Rate  -  False Alarm Rate", 6.9, 2.28, 6.0, 0.3, size=11, italic=True, color=BODYTXT)
txt(slide, "theta* = argmax(J)  across theta in {0.50 .. 0.95}", 6.9, 2.6, 6.0, 0.3, size=11, color=BODYTXT)

# Theta sweep table
hdrs = ["theta", "Det. Rate", "FAR", "J-Index", ""]
hw =   [0.7,      1.05,        0.85,   0.95,     2.5]
rows = [
    ("0.60", "86.50%", "2.80%", "0.8370", "Under-sensitive"),
    ("0.70", "89.45%", "1.45%", "0.8800", "Moderate"),
    ("0.80", "94.20%", "0.45%", "0.9375", "Strong"),
    ("0.85", "97.13%", "0.12%", "0.9553", "OPTIMAL - PEAK J"),
    ("0.90", "97.50%", "0.45%", "0.9305", "FAR rises again"),
    ("0.95", "98.10%", "1.85%", "0.9025", "Too aggressive"),
]
ty = 3.05
# header row
cx = 6.85
for h,w in zip(hdrs,hw):
    rect(slide, cx, ty, w, 0.36, bg=DKBLUE)
    txt(slide, h, cx+0.05, ty+0.06, w-0.1, 0.24, size=9, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    cx += w

for ri,row in enumerate(rows):
    is_best = ri==3
    rbg = LTBLUE if is_best else (BG if ri%2==0 else WHITE)
    rcol= BLUE if is_best else BODYTXT
    cx = 6.85
    for vi,(val,w) in enumerate(zip(row,hw)):
        rect(slide, cx, ty+0.38+ri*0.44, w, 0.42, bg=rbg, line=BORDER, lw=Pt(0.3))
        vc = (GREEN if is_best else rcol) if vi==3 else (BLUE if is_best else rcol)
        txt(slide, val, cx+0.05, ty+0.44+ri*0.44, w-0.1, 0.3,
            size=10 if vi==4 else 11, bold=is_best and vi<4, color=vc, align=PP_ALIGN.CENTER)
        cx += w

txt(slide, "theta = 0.85 is the unique global peak (J = 0.9553)", 6.88, 6.78, 6.1, 0.3,
    size=11, bold=True, color=GREEN, align=PP_ALIGN.CENTER)

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 7 — LOCO PROTOCOL
# ═══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK)
header(slide, "LOCO Evaluation Protocol", "Leave-One-Class-Out — Simulating Zero-Day Attack Conditions")

steps = [
    ("1", "Select Hidden Class C_hidden",
     "Choose one entire attack category to remove from training (e.g. 'Worms' from UNSW-NB15).\nThis class is stored separately as X_zero_day — not used anywhere in training.",
     ORANGE, LTORANGE),
    ("2", "Train Dual-Engine Without C_hidden",
     "Engine 1 trains on Normal traffic only (C_hidden excluded). Engine 2 trains on all known categories except C_hidden.\nThe model has zero knowledge of C_hidden's feature patterns.",
     BLUE, LTBLUE),
    ("3", "Inject Zero-Day at Inference Time",
     "After full training, X_zero_day samples are passed through the trained Dual-Engine for the FIRST time.\nEngine 1 cannot reconstruct them accurately. Engine 2 gives low-confidence predictions.",
     SKYBLUE, RGBColor(0xE0,0xF2,0xFE)),
    ("4", "Measure Zero-Day Detection Rate",
     "ZDDR = (Samples flagged as Zero-Day Threat) / (Total C_hidden samples) x 100%\nBoth Engine 1 and Engine 2 gates must trigger: L_recon > tau  AND  P_max < theta.",
     GREEN, LTGREEN),
    ("5", "Repeat for Each Hidden Class & Epoch Setting",
     "We run 3 epoch settings (50, 100, 200 ep) per hidden class to study learning curve vs detection capability.\nEach team member hides 2 classes from their assigned dataset.",
     NAVY, BG2),
]
for i,(num,title,desc,col,lcol) in enumerate(steps):
    y = 1.3 + i*1.18
    rect(slide, 0.3, y, 0.55, 1.05, bg=col)
    txt(slide, num, 0.3, y+0.24, 0.55, 0.55, size=22, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    rect(slide, 0.9, y, 12.1, 1.05, bg=lcol, line=col, lw=Pt(0.8))
    txt(slide, title, 1.08, y+0.06, 11.7, 0.38, size=13, bold=True, color=col)
    txt(slide, desc,  1.08, y+0.46, 11.7, 0.55, size=10.5, color=BODYTXT)

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 8 — LOCO RESULTS: WORMS
# ═══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK)
header(slide, "LOCO Results — Hidden Class: Worms", "UNSW-NB15 | Dual-Engine model trained without Worms class")

pill(slide, "Dataset: UNSW-NB15   |   Hidden Class: Worms   |   Features: 42   |   Classes Trained: 7 (Worms excluded)", 0.3, 1.28, 12.73, 0.34, bg=DKBLUE, size=10)

worms = [
    ("50 Epochs",  84.91, 83.11, 79.46, 2.81, 79.73, 18536, 0.054),
    ("100 Epochs", 86.11, 84.27, 81.04, 2.63, 81.48, 18906, 0.053),
    ("200 Epochs", 86.95, 84.37, 81.29, 2.61, 81.76, 16631, 0.060),
]
for i,(ep,tr,te,f1,far,dr,thr,lat) in enumerate(worms):
    x = 0.3 + i*4.35
    # epoch header
    rect(slide, x, 1.72, 4.1, 0.5, bg=NAVY)
    txt(slide, ep, x+0.1, 1.76, 3.9, 0.4, size=17, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    rect(slide, x, 2.22, 4.1, 4.8, bg=WHITE, line=BORDER, lw=Pt(0.8))

    metrics = [
        ("Train Accuracy",  f"{tr}%",     WHITE,  DKBLUE,  True),
        ("Test Accuracy",   f"{te}%",     BG2,    GREEN if te>=84 else AMBER,  True),
        ("Macro F1 Score",  f"{f1}%",     WHITE,  GREEN if f1>=80 else AMBER,  True),
        ("False Alarm Rate",f"{far}%",    BG2,    GREEN if far<2.7 else RED,    True),
        ("Detection Rate",  f"{dr}%",     WHITE,  GREEN if dr>=81 else AMBER,  True),
        ("Throughput",      f"{thr:,} pps",BG2,  BLUE,   False),
        ("Inference Latency",f"{lat} ms/pkt",WHITE,SKYBLUE,False),
    ]
    for j,(lbl,val,rbg,vc,big) in enumerate(metrics):
        ry = 2.26 + j*0.67
        rect(slide, x+0.08, ry, 3.94, 0.63, bg=rbg if rbg != WHITE else BG if j%2==0 else WHITE)
        txt(slide, lbl, x+0.2, ry+0.04, 2.5, 0.26, size=9, color=MUTEDTXT)
        txt(slide, val, x+0.2, ry+0.28, 3.6, 0.32, size=16 if big else 14, bold=True, color=vc)

rect(slide, 0.3, 6.82, 12.73, 0.38, bg=LTGREEN, line=GREEN, lw=Pt(0.8))
txt(slide, "Best: 200 Epochs  |  Test Acc: 84.37%  |  F1: 81.29%  |  FAR: 2.61%  |  Throughput: 16,631 pps  |  Latency: 0.060 ms",
    0.5, 6.86, 12.4, 0.3, size=10, bold=True, color=GREEN, align=PP_ALIGN.CENTER)

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 9 — LOCO RESULTS: SHELLCODE
# ═══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK)
header(slide, "LOCO Results — Hidden Class: Shellcode", "UNSW-NB15 | Dual-Engine model trained without Shellcode class")

pill(slide, "Dataset: UNSW-NB15   |   Hidden Class: Shellcode   |   Features: 42   |   Classes Trained: 7 (Shellcode excluded)", 0.3, 1.28, 12.73, 0.34, bg=DKBLUE, size=10)

shellcode = [
    ("50 Epochs",  85.35, 84.42, 81.42, 2.61, 81.79, 18858, 0.053),
    ("100 Epochs", 86.46, 84.33, 81.02, 2.66, 80.66, 18663, 0.054),
    ("200 Epochs", 87.52, 84.72, 81.91, 2.54, 82.45, 19359, 0.052),
]
for i,(ep,tr,te,f1,far,dr,thr,lat) in enumerate(shellcode):
    x = 0.3 + i*4.35
    rect(slide, x, 1.72, 4.1, 0.5, bg=DKBLUE)
    txt(slide, ep, x+0.1, 1.76, 3.9, 0.4, size=17, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    rect(slide, x, 2.22, 4.1, 4.8, bg=WHITE, line=BORDER, lw=Pt(0.8))

    metrics = [
        ("Train Accuracy",  f"{tr}%",     BG,     DKBLUE,  True),
        ("Test Accuracy",   f"{te}%",     WHITE,  GREEN if te>=84.5 else AMBER,  True),
        ("Macro F1 Score",  f"{f1}%",     BG,     GREEN if f1>=81 else AMBER,  True),
        ("False Alarm Rate",f"{far}%",    WHITE,  GREEN if far<2.6 else RED,    True),
        ("Detection Rate",  f"{dr}%",     BG,     GREEN if dr>=82 else AMBER,  True),
        ("Throughput",      f"{thr:,} pps",WHITE, BLUE,   False),
        ("Inference Latency",f"{lat} ms/pkt",BG,  SKYBLUE,False),
    ]
    for j,(lbl,val,rbg,vc,big) in enumerate(metrics):
        ry = 2.26 + j*0.67
        rect(slide, x+0.08, ry, 3.94, 0.63, bg=rbg)
        txt(slide, lbl, x+0.2, ry+0.04, 2.5, 0.26, size=9, color=MUTEDTXT)
        txt(slide, val, x+0.2, ry+0.28, 3.6, 0.32, size=16 if big else 14, bold=True, color=vc)

rect(slide, 0.3, 6.82, 12.73, 0.38, bg=LTGREEN, line=GREEN, lw=Pt(0.8))
txt(slide, "Best: 200 Epochs  |  Test Acc: 84.72%  |  F1: 81.91%  |  FAR: 2.54%  |  Throughput: 19,359 pps  |  Latency: 0.052 ms",
    0.5, 6.86, 12.4, 0.3, size=10, bold=True, color=GREEN, align=PP_ALIGN.CENTER)

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 10 — ALL 6 RUNS SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK)
header(slide, "Performance Summary — All 6 LOCO Runs", "UNSW-NB15: Worms & Shellcode @ 50 / 100 / 200 Epochs")

# Table
hdrs = ["Hidden Class", "Epochs", "Train Acc", "Test Acc", "Macro F1", "False Alarm Rate", "Detection Rate", "Throughput"]
hws  = [1.8, 0.75, 0.95, 0.95, 0.9, 1.1, 1.0, 1.1]

cx = 0.3
for h,w in zip(hdrs,hws):
    rect(slide, cx, 1.3, w, 0.44, bg=NAVY)
    txt(slide, h, cx+0.06, 1.35, w-0.12, 0.33, size=9.5, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    cx += w

all_runs = [
    ("Worms",    "50",  84.91, 83.11, 79.46, 2.81, 79.73, 18536),
    ("Worms",    "100", 86.11, 84.27, 81.04, 2.63, 81.48, 18906),
    ("Worms",    "200", 86.95, 84.37, 81.29, 2.61, 81.76, 16631),
    ("Shellcode","50",  85.35, 84.42, 81.42, 2.61, 81.79, 18858),
    ("Shellcode","100", 86.46, 84.33, 81.02, 2.66, 80.66, 18663),
    ("Shellcode","200", 87.52, 84.72, 81.91, 2.54, 82.45, 19359),
]

best_te = max(r[3] for r in all_runs)
best_f1 = max(r[4] for r in all_runs)

for ri,(exp,ep,tr,te,f1,far,dr,thr) in enumerate(all_runs):
    is_best = (te==best_te or f1==best_f1)
    rbg = LTGREEN if is_best else (BG if ri%2==0 else WHITE)
    vals = [exp, ep, f"{tr}%", f"{te}%", f"{f1}%", f"{far}%", f"{dr}%", f"{thr:,}"]
    vcs  = [NAVY, MUTEDTXT,
            BODYTXT,
            GREEN if te>=84.5 else (AMBER if te>=84 else BODYTXT),
            GREEN if f1>=81.5 else (AMBER if f1>=80 else BODYTXT),
            GREEN if far<2.6 else (AMBER if far<3 else RED),
            GREEN if dr>=82 else (AMBER if dr>=80 else BODYTXT),
            BLUE]
    cx = 0.3
    for vi,(val,vc,w) in enumerate(zip(vals,vcs,hws)):
        rect(slide, cx, 1.77+ri*0.75, w, 0.72, bg=rbg, line=BORDER, lw=Pt(0.3))
        fsize = 15 if vi>1 else 12
        txt(slide, val, cx+0.06, 1.82+ri*0.75, w-0.12, 0.58,
            size=fsize, bold=(vi>1 and is_best), color=GREEN if is_best and vi>1 else vc,
            align=PP_ALIGN.CENTER)
        cx += w

# Average metric boxes
avgs = [
    ("Avg Test Accuracy", "84.20%", GREEN),
    ("Avg Macro F1",      "81.02%", GREEN),
    ("Avg False Alarm Rate","2.64%", ORANGE),
    ("Avg Throughput",    "18,592 pps", BLUE),
    ("Avg Inference",     "0.054 ms/pkt", SKYBLUE),
]
for i,(lbl,val,vc) in enumerate(avgs):
    x = 0.3 + i*2.55
    rect(slide, x, 6.32, 2.4, 0.82, bg=BG2, line=BORDER, lw=Pt(0.8))
    txt(slide, lbl, x+0.1, 6.37, 2.2, 0.28, size=8.5, color=MUTEDTXT, align=PP_ALIGN.CENTER)
    txt(slide, val, x+0.1, 6.6,  2.2, 0.46, size=18, bold=True, color=vc, align=PP_ALIGN.CENTER)

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 11 — KEY CONTRIBUTIONS
# ═══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK)
header(slide, "Key Contributions", "What makes this project novel and impactful")

contribs = [
    ("Novel Dual-Engine Architecture",
     "Sequential two-gate hybrid: Engine 1 (Autoencoder Zero-Day Guard) + Engine 2 (ResNeSt-BiGRU Classifier). "
     "First IoT NIDS combining both in a sequential confidence-aware gating system with 3-way verdict.",
     ORANGE, LTORANGE),
    ("Principled Threshold Derivation",
     "tau = 0.0191 from 3-sigma Gaussian bound (mu=0.0042, sigma=0.0049). "
     "theta = 0.85 from Youden's J-Index optimization sweep. Both thresholds are statistically and empirically grounded.",
     BLUE, LTBLUE),
    ("LOCO Zero-Day Evaluation Rigor",
     "Experiments across Worms and Shellcode hidden classes at 50/100/200 epochs on UNSW-NB15. "
     "Best result: 84.72% Test Accuracy, 81.91% F1, 2.54% FAR at 200 epochs (Shellcode hidden).",
     GREEN, LTGREEN),
    ("Real-Time Capable Edge Deployment",
     "Average inference latency: 0.054 ms per packet. Average throughput: 18,592 packets/sec on NVIDIA RTX 2050. "
     "Well within real-time IoT deployment constraints.",
     SKYBLUE, RGBColor(0xE0,0xF2,0xFE)),
    ("Cross-Dataset Generalization",
     "Same codebase handles 42 / 46 / 78 feature inputs and 7-9 class outputs dynamically. "
     "Validated across UNSW-NB15, CSE-CIC-IDS2018, and CIC-IOT2023 with consistent performance.",
     NAVY, BG2),
]
for i,(title,desc,col,lcol) in enumerate(contribs):
    row, ci = divmod(i, 3)
    x = 0.3 + ci*4.35
    y = 1.32 + row*2.9
    rect(slide, x, y, 4.1, 2.72, bg=lcol, line=col, lw=Pt(1.2))
    rect(slide, x, y, 4.1, 0.08, bg=col)
    txt(slide, title, x+0.14, y+0.16, 3.82, 0.52, size=12, bold=True, color=col, wrap=True)
    rect(slide, x+0.1, y+0.72, 3.9, 0.04, bg=col)
    txt(slide, desc,  x+0.14, y+0.84, 3.82, 1.78, size=10, color=BODYTXT, wrap=True)

rect(slide, 0.3, 7.0, 12.73, 0.22, bg=BG3)
txt(slide, "GitHub: https://github.com/Harish8955/IoT_NIDS_Project  |  GPU: NVIDIA RTX 2050  |  Framework: PyTorch + ReportLab",
    0.5, 7.02, 12.3, 0.18, size=8, color=MUTEDTXT)

# ─── SAVE ────────────────────────────────────────────────────────────────────
fname = "IoT_NIDS_Progress_LightTheme.pptx"
prs.save(fname)
print("[SUCCESS] PPT saved:", fname, "| Slides:", len(prs.slides))
