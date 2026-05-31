"""Build the 3-minute pitch deck (.pptx) for Industrial AI Track 1.

Structure follows the official slide spec (max 10 slides):
  1  team + one-sentence what you built
  1  problem + why it matters
  3  approach + key technical decisions
  3  results, metrics, evidence
  1  what you'd do next
All numbers come from results/*/metrics/*.json on the feat/transformer branch.
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

NAVY = RGBColor(0x0B, 0x2A, 0x4A)
NAVY_2 = RGBColor(0x12, 0x3A, 0x63)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GOLD = RGBColor(0xD4, 0xAF, 0x37)
MIST = RGBColor(0xB9, 0xC7, 0xD6)
GREEN = RGBColor(0x4C, 0xC9, 0x8A)

SW, SH = Inches(13.333), Inches(7.5)

prs = Presentation()
prs.slide_width = SW
prs.slide_height = SH
BLANK = prs.slide_layouts[6]


def bg(slide, color=NAVY):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color


def rect(slide, x, y, w, h, color):
    from pptx.enum.shapes import MSO_SHAPE
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def txt(slide, x, y, w, h, runs, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
        space_after=6, line_spacing=1.05):
    """runs: list of paragraphs; each paragraph is list of (text,size,color,bold,italic)."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for i, para in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        p.space_before = Pt(0)
        p.line_spacing = line_spacing
        for (t, size, color, bold, italic) in para:
            r = p.add_run()
            r.text = t
            r.font.size = Pt(size)
            r.font.color.rgb = color
            r.font.bold = bold
            r.font.italic = italic
            r.font.name = "Calibri"
    return tb


def header(slide, kicker, title):
    rect(slide, 0, Inches(1.55), Inches(0.55), Inches(0.06), GOLD)
    txt(slide, Inches(0.7), Inches(0.5), Inches(12), Inches(0.5),
        [[(kicker, 14, GOLD, True, False)]])
    txt(slide, Inches(0.7), Inches(0.85), Inches(12), Inches(0.9),
        [[(title, 30, WHITE, True, False)]])


def bullets(slide, items, x=Inches(0.7), y=Inches(1.9), w=Inches(12), h=Inches(5.2),
            size=17, gap=10):
    paras = []
    for it in items:
        if isinstance(it, tuple):
            lead, rest = it
            paras.append([("‣  ", size, GOLD, True, False),
                          (lead, size, WHITE, True, False),
                          (rest, size, MIST, False, False)])
        else:
            paras.append([("‣  ", size, GOLD, True, False),
                          (it, size, MIST, False, False)])
    txt(slide, x, y, w, h, paras, space_after=gap, line_spacing=1.08)


# ---------------------------------------------------------------- Slide 1: Team
s = prs.slides.add_slide(BLANK); bg(s)
rect(s, 0, 0, Inches(0.22), SH, GOLD)
txt(s, Inches(0.9), Inches(1.1), Inches(11.5), Inches(0.5),
    [[("INDUSTRIAL AI · TRACK 1 — LEARNING & BENCHMARKING PROCESS LOGIC", 14, GOLD, True, False)]])
txt(s, Inches(0.9), Inches(1.7), Inches(11.6), Inches(1.8),
    [[("Small Transformers for", 44, WHITE, True, False)],
     [("Semiconductor Process Logic", 44, WHITE, True, False)]], line_spacing=1.0)
rect(s, Inches(0.95), Inches(3.55), Inches(0.8), Inches(0.06), GOLD)
txt(s, Inches(0.9), Inches(3.8), Inches(11.6), Inches(1.4),
    [[("We trained a compact, family-conditioned GPT decoder that learns fab "
       "process-sequence logic from synthetic data — and benchmarked it against an "
       "n-gram baseline to show it understands process rules rather than memorising them.",
       20, WHITE, False, False)]], line_spacing=1.15)
txt(s, Inches(0.9), Inches(6.2), Inches(11.6), Inches(0.9),
    [[("Team: ", 16, GOLD, True, False), ("[Team Name]", 16, WHITE, True, False),
      ("   ·   Fedja Bogataj", 16, MIST, False, False)],
     [("zero_one Hackathon   ·   Mentor: Simeon (Infineon)   ·   "
       "PyTorch · sentence-transformers · SLURM (Leonardo) · W&B", 13, MIST, False, False)]],
    space_after=4)

# ------------------------------------------------------------- Slide 2: Problem
s = prs.slides.add_slide(BLANK); bg(s)
header(s, "THE PROBLEM", "Do models learn process logic, or just memorise patterns?")
bullets(s, [
    ("Semiconductor manufacturing is a strict recipe. ",
     "Each wafer lot runs ~110–150 ordered steps from a vocabulary of ~120 step types, "
     "with a fixed block structure and 10 hard ordering rules (e.g. no etch without a "
     "prior litho develop, no test before passivation cure)."),
    ("Three product families. ",
     "MOSFET (~126 steps), IGBT (~151), IC (~107) — each with its own prep blocks and "
     "litho-cycle counts, but a shared process backbone."),
    ("A model can fake it. ",
     "High next-step accuracy is easy from local frequencies. The hard question is whether "
     "a model captures long-range ordering constraints — the actual process logic."),
    ("Why it matters. ",
     "A held-out 4th family tests out-of-distribution generalisation. Getting this right means "
     "a sovereign, open, reproducible model of fab logic — not a black-box API wrapper."),
], size=17, gap=14)

# ----------------------------------------------------- Slide 3: Approach — setup
s = prs.slides.add_slide(BLANK); bg(s)
header(s, "APPROACH · 1 OF 3", "Frame it as language modelling over process steps")
bullets(s, [
    ("One step = one token. ",
     "Sequences always start RECEIVE WAFER LOT and end SHIP LOT. We model the whole "
     "sequence autoregressively, conditioned on the product family."),
    ("Data — synthetic but grammar-aware. ",
     "On top of the 1,000 canonical variants per family, we generated 5k / 10k / 25k extra "
     "sequences with the validator-backed generator. All 10 forbidden patterns are enforced, "
     "so training data is guaranteed clean."),
    ("Three benchmark tasks. ",
     "Next-step prediction (Top-k, MRR) · Sequence completion (edit distance, token & block "
     "accuracy) · Anomaly detection (F1, ROC-AUC, rule attribution)."),
    ("Anomaly is the real test. ",
     "Detecting a rule violation requires understanding the global ordering grammar — not just "
     "what token tends to follow what."),
], size=17, gap=13)

# --------------------------------------------- Slide 4: Approach — architecture
s = prs.slides.add_slide(BLANK); bg(s)
header(s, "APPROACH · 2 OF 3", "A small GPT decoder, conditioned on the family")
bullets(s, [
    ("Compact causal transformer. ",
     "4 layers · 6 heads · d_model 384 · ff_dim 1536 · Pre-LN + GELU · dropout 0.1 · "
     "max sequence length 256. ~29 MB checkpoint — fast to iterate on a single GPU."),
    ("Family conditioning. ",
     "A learnable family embedding is summed into every position, so the same network "
     "specialises per family without separate models."),
    ("Weight-tied LM head. ",
     "The output projection shares the token-embedding matrix, halving parameters and "
     "regularising the small model."),
    ("Drop-in design. ",
     "The predictor exposes exactly the n-gram baseline's interface (top_k, log_prob), so "
     "the same evaluation pipeline scores both models — a fair, apples-to-apples comparison."),
], size=17, gap=13)

# ------------------------------- Slide 5: Approach — key technical decisions
s = prs.slides.add_slide(BLANK); bg(s)
header(s, "APPROACH · 3 OF 3", "Key decision: semantic embedding init for OOD")
bullets(s, [
    ("The generalisation trick. ",
     "Token embeddings are initialised from sentence-transformers all-MiniLM-L6-v2 (384-dim), "
     "encoding each step as its name + description + parameters."),
    ("Why it helps the hidden 4th family. ",
     "An unseen step from a new family lands in the same semantic space as known steps, so the "
     "model can place it sensibly instead of treating it as a random unknown token."),
    ("Reproducible training on Leonardo. ",
     "AdamW + cosine LR · SLURM sbatch · Weights & Biases tracking with auto-named runs, "
     "score logging, periodic checkpointing and auto-resume."),
    ("Engineered for fast eval. ",
     "bf16 batched inference made evaluation ~10–30× faster, which let us run a full "
     "5k / 10k / 25k scaling study within the hackathon."),
], size=17, gap=13)

# ------------------------------------------------ Slide 6: Results — headline
s = prs.slides.add_slide(BLANK); bg(s)
header(s, "RESULTS · 1 OF 3", "Transformer beats the n-gram on structure")
# table
rows = [
    ("Metric (higher better unless noted)", "N-gram (o3)", "Transformer 5k", "Δ"),
    ("Top-1 next-step accuracy", "0.684", "0.709", "+3.7%"),
    ("MRR (next-step)", "0.837", "0.852", "+1.8%"),
    ("Token accuracy (completion)", "0.254", "0.401", "+57%"),
    ("Block accuracy (completion)", "0.479", "0.648", "+35%"),
    ("Norm. edit distance ↓ (lower better)", "0.562", "0.217", "−61%"),
    ("Anomaly ROC-AUC", "0.515", "0.794", "+54%"),
]
tx, ty = Inches(0.7), Inches(1.95)
colw = [Inches(5.6), Inches(2.2), Inches(2.6), Inches(1.7)]
rh = Inches(0.6)
cx = tx
for j, w in enumerate(colw):
    rect(s, cx, ty, w, rh, NAVY_2)
    cx += w
# header row text
cx = tx
for j, (w) in enumerate(colw):
    head = rows[0][j]
    txt(s, cx + Inches(0.1), ty, w - Inches(0.2), rh,
        [[(head, 13, GOLD, True, False)]], anchor=MSO_ANCHOR.MIDDLE,
        align=PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER)
    cx += w
for i in range(1, len(rows)):
    ry = ty + rh * i
    cx = tx
    band = NAVY if i % 2 else NAVY_2
    for j, w in enumerate(colw):
        rect(s, cx, ry, w, rh, band)
        cx += w
    cx = tx
    for j, w in enumerate(colw):
        val = rows[i][j]
        col = WHITE
        if j == 2:
            col = WHITE
        if j == 3:
            col = GREEN
        if j == 1:
            col = MIST
        txt(s, cx + Inches(0.1), ry, w - Inches(0.2), rh,
            [[(val, 14, col, j in (2, 3), False)]], anchor=MSO_ANCHOR.MIDDLE,
            align=PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER)
        cx += w
txt(s, Inches(0.7), Inches(6.7), Inches(12), Inches(0.6),
    [[("Identical eval set: n = 6,000 (Tasks 1–2), n = 3,000 (Task 3). "
       "Biggest gains are exactly on the structural metrics — completion and anomaly.",
       13, MIST, False, True)]])

# ----------------------------------------- Slide 7: Results — anomaly signal
s = prs.slides.add_slide(BLANK); bg(s)
header(s, "RESULTS · 2 OF 3", "Anomaly detection: where understanding shows up")
bullets(s, [
    ("Both models score F1 = 1.0 — and that's a trap. ",
     "The submission can call the rule-based validator, so binary accuracy is perfect for "
     "anyone. F1 alone proves nothing about the model."),
    ("ROC-AUC on the model's own confidence is the honest signal. ",
     "It asks: does the model itself rank invalid sequences as less likely? n-gram = 0.515 "
     "(essentially a coin flip). Transformer = 0.794 → 0.839 → 0.850 as data grows."),
    ("Rule attribution. ",
     "97.5% accuracy at naming which of the 10 rules was violated — useful, explainable output "
     "for a process engineer."),
    ("Takeaway. ",
     "The transformer has internalised the ordering grammar; the n-gram has not. That gap is "
     "the whole point of the benchmark."),
], size=17, gap=13)
txt(s, Inches(0.7), Inches(6.75), Inches(12), Inches(0.5),
    [[("Confusion matrix (n=3,000): TP 1170 · TN 1830 · FP 0 · FN 0 — validator-backed labels.",
       13, MIST, False, True)]])

# ------------------------------------------- Slide 8: Results — scaling study
s = prs.slides.add_slide(BLANK); bg(s)
header(s, "RESULTS · 3 OF 3", "Scaling study: 5k → 10k → 25k sequences")
rows = [
    ("Metric", "5k", "10k", "25k"),
    ("Top-1 next-step", "0.709", "0.690", "0.703"),
    ("MRR", "0.852", "0.842", "0.849"),
    ("Token accuracy", "0.401", "0.393", "0.393"),
    ("Block accuracy", "0.648", "0.645", "0.643"),
    ("Anomaly ROC-AUC", "0.794", "0.839", "0.850"),
]
tx, ty = Inches(0.7), Inches(1.95)
colw = [Inches(5.0), Inches(2.0), Inches(2.0), Inches(2.0)]
rh = Inches(0.55)
for i in range(len(rows)):
    ry = ty + rh * i
    cx = tx
    band = NAVY_2 if i == 0 else (NAVY if i % 2 else NAVY_2)
    for j, w in enumerate(colw):
        rect(s, cx, ry, w, rh, band)
        head = rows[i][j]
        col = GOLD if i == 0 else (WHITE if j == 0 else MIST)
        txt(s, cx + Inches(0.1), ry, w - Inches(0.2), rh,
            [[(head, 13 if i == 0 else 14, col, i == 0 or j == 0, False)]],
            anchor=MSO_ANCHOR.MIDDLE,
            align=PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER)
        cx += w
txt(s, Inches(0.7), Inches(5.45), Inches(12), Inches(1.6),
    [[("Two honest findings:", 17, GOLD, True, False)],
     [("‣  ", 16, GOLD, True, False),
      ("Prediction & completion plateau at 5k", 16, WHITE, True, False),
      (" — the synthetic data is too clean, so the small model saturates almost immediately.",
       16, MIST, False, False)],
     [("‣  ", 16, GOLD, True, False),
      ("Anomaly ROC-AUC keeps climbing with data", 16, WHITE, True, False),
      (" (0.79 → 0.85) — evidence it is still learning structure, not memorising sequences.",
       16, MIST, False, False)]], space_after=10, line_spacing=1.1)

# ---------------------------------------------------- Slide 9: What's next
s = prs.slides.add_slide(BLANK); bg(s)
header(s, "WHAT WE'D DO NEXT", "From benchmark to genuine generalisation")
bullets(s, [
    ("Evaluate on the hidden 4th family. ",
     "The real OOD test — and the direct payoff of the semantic-init design. We expect a "
     "smaller performance drop than a model with random embeddings."),
    ("Sub-word tokenizer experiment. ",
     "In progress on feat/subword-tokenizer: break step strings into sub-word units so the "
     "model shares structure across related, never-before-seen steps."),
    ("Harder, dirtier data. ",
     "Inject controlled near-miss violations so the prediction tasks stop saturating and the "
     "scaling curve has room to move."),
    ("Per-rule and per-family breakdowns. ",
     "Report which of the 10 rules and which families are hardest — turning the benchmark into "
     "an actionable diagnostic for process engineers."),
], size=17, gap=13)
txt(s, Inches(0.7), Inches(6.75), Inches(12), Inches(0.5),
    [[("Working artifact: full code, trained 5k/10k/25k checkpoints, and reproducible "
       "results/ tables on branch feat/transformer.", 13, GOLD, False, True)]])

out = "Small_Transformers_Process_Logic.pptx"
prs.save(out)
print("saved", out, "·", len(prs.slides.__iter__.__self__._sldIdLst), "slides")
