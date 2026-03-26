import os
import re
import tkinter as tk
from tkinter import filedialog, font, messagebox
from collections import Counter, defaultdict

# -> We use NLTK for real tokenization and standard corpora,
# so the dataset is larger and the predictions look realistic.
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize


# -> App-level settings for title, dataset size limits, and UI colors.
APP_TITLE = "Predictive Text Generator"
MAX_TOKENS = 200000
MAX_SOURCE_CHARS = 160000

COLORS = {
    "outer_bg": "#0b1220",
    "phone_body": "#111827",
    "screen_bg": "#f8fafc",
    "header": "#ff7a00",
    "header_dark": "#e86a00",
    "header_light": "#ffb570",
    "text": "#0f172a",
    "muted": "#64748b",
    "border": "#e2e8f0",
    "chip_bg": "#e2e8f0",
    "chip_active": "#cbd5e1",
    "primary_btn": "#0f172a",
    "primary_btn_active": "#1f2937",
    "secondary_btn": "#e2e8f0",
    "secondary_btn_active": "#cbd5e1",
    "keyboard_bg": "#f1f5f9",
    "key_bg": "#e2e8f0",
    "key_active": "#cbd5e1",
}

DEFAULT_SAMPLE_TEXT = """
Predictive text suggests the next word based on context and patterns.
Natural language processing is a key part of modern AI applications.
This demo uses an N-gram model trained on multiple corpora.
More data improves the quality and variety of the suggestions.
"""

# -> We keep only word tokens (letters + apostrophes) for clean modeling.
WORD_RE = re.compile(r"[a-z']+")


# -> Safe file reader to cap large text files during demo.
def _read_file(path, limit_chars=MAX_SOURCE_CHARS):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(limit_chars)
    except Exception:
        return ""


# -> Verify required NLTK resources; fail fast with a clear command.
def ensure_nltk_data():
    required = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "gutenberg": "corpora/gutenberg",
        "brown": "corpora/brown",
        "reuters": "corpora/reuters",
    }

    def _has_resource(path):
        try:
            nltk.data.find(path)
            return True
        except LookupError:
            try:
                nltk.data.find(path + ".zip")
                return True
            except LookupError:
                return False

    downloaded = []
    for pkg, path in required.items():
        if not _has_resource(path):
            try:
                nltk.download(pkg, quiet=True)
                downloaded.append(pkg)
            except Exception:
                pass
    missing = []
    for pkg, path in required.items():
        if not _has_resource(path):
            missing.append(pkg)
    if missing:
        raise RuntimeError(
            "NLTK data missing. Run: python -m nltk.downloader "
            + " ".join(missing)
        )
    return downloaded


# -> Load text from local file + NLTK corpora to expand the dataset.
def load_text_sources(base_dir):
    sources = []
    local_path = os.path.join(base_dir, "data.txt")
    if os.path.exists(local_path):
        sources.append(("data.txt", _read_file(local_path)))

    from nltk.corpus import gutenberg, brown, reuters

    gutenberg_ids = [
        "austen-emma.txt",
        "austen-persuasion.txt",
        "melville-moby_dick.txt",
    ]
    gutenberg_text = "".join(gutenberg.raw(fid) for fid in gutenberg_ids)
    sources.append(("gutenberg", gutenberg_text[:MAX_SOURCE_CHARS]))

    brown_text = brown.raw(categories=["news", "editorial", "reviews"])
    sources.append(("brown", brown_text[:MAX_SOURCE_CHARS]))

    reuters_text = reuters.raw(categories=["trade", "crude", "wheat"])
    sources.append(("reuters", reuters_text[:MAX_SOURCE_CHARS]))

    if not sources:
        sources.append(("sample", DEFAULT_SAMPLE_TEXT))
    return sources


class NgramPredictor:
    def __init__(self, n=3):
        # -> We keep unigram, bigram, and trigram counts for fast lookup.
        self.n = max(2, int(n))
        self.models = {k: defaultdict(Counter) for k in range(2, self.n + 1)}
        self.unigrams = Counter()
        self.vocab = set()
        self.total_tokens = 0

    def tokenize(self, text):
        # -> Tokenize, lowercase, and filter to consistent word tokens.
        try:
            tokens = word_tokenize(text.lower())
        except LookupError as exc:
            raise RuntimeError(
                "NLTK tokenizer data missing. Run: python -m nltk.downloader punkt punkt_tab"
            ) from exc
        return [t for t in tokens if WORD_RE.fullmatch(t)]

    def train_tokens(self, tokens):
        # -> Build frequency tables for all n-grams.
        if not tokens:
            return
        self.total_tokens += len(tokens)
        self.vocab.update(tokens)
        self.unigrams.update(tokens)
        for n in range(2, self.n + 1):
            for gram in ngrams(tokens, n):
                context = gram[:-1]
                nxt = gram[-1]
                self.models[n][context][nxt] += 1

    def train_from_texts(self, texts, max_tokens=MAX_TOKENS):
        # -> Combine text sources, limit total size, then train once.
        tokens = []
        for text in texts:
            if not text:
                continue
            tokens.extend(self.tokenize(text))
            if max_tokens and len(tokens) >= max_tokens:
                tokens = tokens[:max_tokens]
                break
        self.train_tokens(tokens)

    def _prefix(self, text):
        # -> Detect a partial word so we can autocomplete it.
        match = re.search(r"([a-zA-Z']+)$", text)
        if match and not text.endswith(" "):
            return match.group(1).lower()
        return ""

    def _filter_candidates(self, counter, prefix, k):
        items = counter.most_common()
        if prefix:
            items = [(w, c) for w, c in items if w.startswith(prefix)]
        return [w for w, _ in items[:k]]

    def suggest(self, text, k=3):
        # -> Use longest context first (trigram -> bigram -> unigram).
        prefix = self._prefix(text)
        context_text = text[:-len(prefix)] if prefix else text
        tokens = self.tokenize(context_text)
        for ctx_len in range(min(self.n - 1, len(tokens)), 0, -1):
            context = tuple(tokens[-ctx_len:])
            model = self.models.get(ctx_len + 1)
            if model and context in model:
                return self._filter_candidates(model[context], prefix, k)
        return self._filter_candidates(self.unigrams, prefix, k)

    def complete_with(self, text, word):
        # -> Replace a partial word or append a new word with spacing.
        if not word:
            return text
        match = re.search(r"([a-zA-Z']+)$", text)
        if match and not text.endswith(" "):
            start = match.start(1)
            return text[:start] + word + " "
        sep = "" if not text or text.endswith(" ") else " "
        return text + sep + word + " "


# -> Helper to draw rounded rectangles for the phone UI.
def round_rect(canvas, x1, y1, x2, y2, r=24, **kw):
    points = [
        x1 + r,
        y1,
        x2 - r,
        y1,
        x2,
        y1,
        x2,
        y1 + r,
        x2,
        y2 - r,
        x2,
        y2,
        x2 - r,
        y2,
        x1 + r,
        y2,
        x1,
        y2,
        x1,
        y2 - r,
        x1,
        y1 + r,
        x1,
        y1,
    ]
    return canvas.create_polygon(points, **kw, smooth=True)


class PhoneFrame(tk.Tk):
    def __init__(self, title=APP_TITLE):
        # -> Create a phone-like frame to make the demo feel familiar.
        super().__init__()
        self.title(title)
        self.configure(bg=COLORS["outer_bg"])
        self.resizable(False, False)

        base_w, base_h = 420, 820
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        usable_w = screen_w - 160
        usable_h = screen_h - 220
        scale = min(usable_w / base_w, usable_h / base_h, 1.0)
        scale = max(scale, 0.55)
        self.ui_scale = scale
        self.phone_w, self.phone_h = int(base_w * scale), int(base_h * scale)

        def s(value):
            return int(value * scale)

        win_w = self.phone_w + s(40)
        win_h = self.phone_h + s(40)
        pos_x = max(0, (screen_w - win_w) // 2)
        pos_y = max(0, (screen_h - win_h) // 2)
        self.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")

        self.canvas = tk.Canvas(
            self,
            width=win_w,
            height=win_h,
            highlightthickness=0,
            bg=COLORS["outer_bg"],
        )
        self.canvas.pack(padx=s(10), pady=s(10))

        round_rect(
            self.canvas,
            s(20),
            s(20),
            s(20) + self.phone_w,
            s(20) + self.phone_h,
            r=s(36),
            fill=COLORS["phone_body"],
            outline="",
        )
        round_rect(
            self.canvas,
            s(34),
            s(64),
            s(34) + self.phone_w - s(28),
            s(64) + self.phone_h - s(90),
            r=s(28),
            fill=COLORS["screen_bg"],
            outline="",
        )
        self.canvas.create_oval(
            s(200), s(44), s(280), s(68), fill=COLORS["phone_body"], outline=""
        )

        self.screen_frame = tk.Frame(self.canvas, bg=COLORS["screen_bg"])
        self.canvas.create_window(
            s(40),
            s(70),
            anchor="nw",
            width=self.phone_w - s(40),
            height=self.phone_h - s(102),
            window=self.screen_frame,
        )


class MobilePredictiveApp:
    def __init__(self, root, predictor, sources):
        # -> Wire the model to the UI and enable live suggestions.
        self.root = root
        self.predictor = predictor
        self.sources = sources
        self.shift_on = False
        self.auto_suggest = tk.BooleanVar(value=True)
        self.placeholder = "Type your message..."
        self._build_ui()
        self.refresh_suggestions()

    def _build_ui(self):
        # -> Main UI layout: header, stats, text area, chips, buttons, keyboard.
        sf = self.root.screen_frame

        scale = getattr(self.root, "ui_scale", 1.0)

        def px(value):
            return max(1, int(value * scale))

        self.hfont = font.Font(family="Segoe UI", size=max(12, int(16 * scale)), weight="bold")
        self.sfont = font.Font(family="Segoe UI", size=max(8, int(10 * scale)))
        self.tfont = font.Font(family="Segoe UI", size=max(9, int(12 * scale)))
        self.kfont = font.Font(family="Segoe UI", size=max(9, int(11 * scale)), weight="bold")

        header = tk.Canvas(sf, height=px(110), bg=COLORS["header"], highlightthickness=0)
        header.pack(fill="x")
        header.create_oval(px(-80), px(-40), px(140), px(120), fill=COLORS["header_light"], outline="")
        header.create_oval(px(200), px(-20), px(440), px(140), fill=COLORS["header_dark"], outline="")
        header.create_text(
            px(18),
            px(32),
            text="Predictive Text",
            anchor="w",
            font=self.hfont,
            fill="white",
        )
        header.create_text(
            18,
            px(64),
            text="N-gram model with NLTK corpora",
            anchor="w",
            font=self.sfont,
            fill="white",
        )

        stats = tk.Frame(sf, bg=COLORS["screen_bg"])
        stats.pack(fill="x", padx=px(14), pady=(px(8), px(4)))
        self.stats_label = tk.Label(
            stats,
            text=self._stats_text(),
            bg=COLORS["screen_bg"],
            fg=COLORS["muted"],
            font=self.sfont,
            anchor="w",
        )
        self.stats_label.pack(side="left", fill="x", expand=True)

        data_row = tk.Frame(sf, bg=COLORS["screen_bg"])
        data_row.pack(fill="x", padx=px(14), pady=(0, px(6)))
        add_btn = tk.Button(
            data_row,
            text="Add Data File",
            command=self._add_file,
            bg=COLORS["header"],
            fg="white",
            bd=0,
            padx=px(12),
            pady=px(6),
            font=self.sfont,
            activebackground=COLORS["header_dark"],
            relief="flat",
        )
        add_btn.pack(side="left")

        text_wrap = tk.Frame(
            sf,
            bg="white",
            highlightthickness=1,
            highlightbackground=COLORS["border"],
        )
        text_wrap.pack(fill="both", expand=True, padx=px(14), pady=(px(6), px(6)))
        self.text = tk.Text(
            text_wrap,
            height=max(5, int(8 * scale)),
            wrap="word",
            bd=0,
            relief="flat",
            bg="white",
            fg=COLORS["muted"],
            padx=12,
            pady=px(12),
            font=self.tfont,
        )
        self.text.pack(fill="both", expand=True)
        self.text.insert("1.0", self.placeholder)

        controls = tk.Frame(sf, bg=COLORS["screen_bg"])
        controls.pack(fill="x", padx=px(14), pady=(0, px(6)))
        auto = tk.Checkbutton(
            controls,
            text="Auto suggest",
            variable=self.auto_suggest,
            onvalue=True,
            offvalue=False,
            bg=COLORS["screen_bg"],
            fg=COLORS["muted"],
            activebackground=COLORS["screen_bg"],
            font=self.sfont,
        )
        auto.pack(side="left")

        self.chip_bar = tk.Frame(sf, bg=COLORS["screen_bg"])
        self.chip_bar.pack(fill="x", padx=px(12), pady=(0, px(6)))
        self.chips = []
        for _ in range(3):
            btn = tk.Button(
                self.chip_bar,
                text="-",
                command=lambda w="": self._apply_chip(w),
                bg=COLORS["chip_bg"],
                fg=COLORS["text"],
                bd=0,
                padx=px(14),
                pady=px(6),
                font=self.tfont,
                relief="flat",
                activebackground=COLORS["chip_active"],
            )
            btn.pack(side="left", padx=px(6))
            self.chips.append(btn)

        actions = tk.Frame(sf, bg=COLORS["screen_bg"])
        actions.pack(fill="x", padx=px(14), pady=(0, px(8)))
        self.gen_btn = tk.Button(
            actions,
            text="Suggest",
            command=self.refresh_suggestions,
            bg=COLORS["primary_btn"],
            fg="white",
            bd=0,
            padx=px(14),
            pady=px(10),
            font=self.tfont,
            relief="flat",
            activebackground=COLORS["primary_btn_active"],
        )
        self.gen_btn.pack(side="left", fill="x", expand=True, padx=(0, px(6)))
        clear_btn = tk.Button(
            actions,
            text="Clear",
            command=self._clear_text,
            bg=COLORS["secondary_btn"],
            fg=COLORS["text"],
            bd=0,
            padx=14,
            pady=10,
            font=self.tfont,
            relief="flat",
            activebackground=COLORS["secondary_btn_active"],
        )
        clear_btn.pack(side="left", fill="x", expand=True, padx=(px(6), 0))

        self._build_keyboard(sf)

        self.text.bind("<FocusIn>", self._clear_placeholder)
        self.text.bind("<FocusOut>", self._restore_placeholder)
        self.text.bind("<KeyRelease>", self._on_key_release)

    def _stats_text(self):
        # -> Show dataset size to emphasize training scale.
        source_names = [name for name, _ in self.sources]
        return (
            f"Tokens: {self.predictor.total_tokens:,}  "
            f"Vocab: {len(self.predictor.vocab):,}  "
            f"Sources: {', '.join(source_names)}"
        )

    def _update_stats(self):
        self.stats_label.config(text=self._stats_text())

    def _clear_placeholder(self, event=None):
        # -> UX polish so the textbox feels like a real chat app.
        if self.text.get("1.0", "end-1c") == self.placeholder:
            self.text.delete("1.0", "end")
            self.text.config(fg=COLORS["text"])

    def _restore_placeholder(self, event=None):
        if not self.text.get("1.0", "end-1c").strip():
            self.text.delete("1.0", "end")
            self.text.insert("1.0", self.placeholder)
            self.text.config(fg=COLORS["muted"])

    def _get_text(self):
        content = self.text.get("1.0", "end-1c")
        return "" if content == self.placeholder else content

    def _clear_text(self):
        self.text.delete("1.0", "end")
        self.text.insert("1.0", self.placeholder)
        self.text.config(fg=COLORS["muted"])
        self.refresh_suggestions()

    def _apply_chip(self, word):
        # -> Apply a suggestion and immediately refresh new options.
        if not word or word == "-":
            return
        content = self._get_text()
        new_text = self.predictor.complete_with(content, word)
        self.text.delete("1.0", "end")
        self.text.insert("1.0", new_text)
        self.text.config(fg=COLORS["text"])
        self.refresh_suggestions()

    def refresh_suggestions(self):
        # -> Pull top-3 next-word predictions from the model.
        content = self._get_text()
        suggs = self.predictor.suggest(content, k=3)
        for i, btn in enumerate(self.chips):
            label = suggs[i] if i < len(suggs) else "-"
            btn.config(text=label, command=lambda w=label: self._apply_chip(w))

    def _on_key_release(self, event):
        if self.auto_suggest.get():
            self.refresh_suggestions()

    def _add_file(self):
        # -> Let users expand the dataset live during the demo.
        path = filedialog.askopenfilename(
            title="Add a text file",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        extra_text = _read_file(path, MAX_SOURCE_CHARS)
        if not extra_text:
            messagebox.showwarning("Add File", "Unable to read that file.")
            return
        self.predictor.train_from_texts([extra_text], max_tokens=60000)
        self.sources.append((os.path.basename(path), extra_text))
        self._update_stats()
        self.refresh_suggestions()

    def _build_keyboard(self, parent):
        # -> On-screen keyboard to mimic a mobile typing experience.
        kb = tk.Frame(parent, bg=COLORS["keyboard_bg"])
        kb.pack(fill="x", padx=px(8), pady=(0, px(8)))

        rows = [
            list("qwertyuiop"),
            list("asdfghjkl"),
            ["SHIFT"] + list("zxcvbnm") + ["BKSP"],
            ["123", "SPACE", "ENTER"],
        ]

        self.alpha_buttons = []

        def insert_char(ch):
            if ch == "SPACE":
                self.text.insert("insert", " ")
            elif ch == "ENTER":
                self.text.insert("insert", "\n")
            elif ch == "BKSP":
                try:
                    self.text.delete("insert-1c")
                except tk.TclError:
                    pass
            elif ch == "SHIFT":
                self.shift_on = not self.shift_on
                self._refresh_keyboard()
                return
            elif ch == "123":
                self._popup_digits()
                return
            else:
                char = ch.upper() if self.shift_on else ch.lower()
                self.text.insert("insert", char)
                if self.shift_on:
                    self.shift_on = False
                    self._refresh_keyboard()
            self.text.config(fg=COLORS["text"])
            self.refresh_suggestions()

        for row in rows:
            rowf = tk.Frame(kb, bg=COLORS["keyboard_bg"])
            rowf.pack(pady=px(3))
            for ch in row:
                label = ch.upper() if len(ch) == 1 else ch
                btn = tk.Button(
                    rowf,
                    text=f" {label} ",
                    bd=0,
                    bg=COLORS["key_bg"],
                    fg=COLORS["text"],
                    font=self.kfont,
                    activebackground=COLORS["key_active"],
                    relief="flat",
                    command=lambda c=ch: insert_char(c),
                )
                btn.pack(side="left", padx=px(3), pady=px(1))
                if len(ch) == 1:
                    self.alpha_buttons.append((ch, btn))

    def _refresh_keyboard(self):
        for ch, btn in self.alpha_buttons:
            btn.config(text=f" {ch.upper() if self.shift_on else ch.lower()} ")

    def _popup_digits(self):
        top = tk.Toplevel(self.root)
        top.title("Digits")
        top.configure(bg=COLORS["screen_bg"])
        for i in range(10):
            tk.Button(
                top,
                text=str(i),
                width=4,
                bd=0,
                bg=COLORS["secondary_btn"],
                activebackground=COLORS["secondary_btn_active"],
                command=lambda c=str(i): (
                    self.text.insert("insert", c),
                    top.destroy(),
                    self.refresh_suggestions(),
                ),
            ).grid(row=i // 5, column=i % 5, padx=4, pady=4)


def main():
    # -> Startup = ensure data, load corpora, train model, launch UI.
    ensure_nltk_data()
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()
    sources = load_text_sources(base_dir)
    predictor = NgramPredictor(n=3)
    predictor.train_from_texts([text for _, text in sources], max_tokens=MAX_TOKENS)
    app = PhoneFrame(APP_TITLE)
    MobilePredictiveApp(app, predictor, sources)
    app.mainloop()


if __name__ == "__main__":
    main()
