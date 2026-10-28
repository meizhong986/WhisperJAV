"""
1.10.0, item 6 (#448; owner W283): the GUI in Simplified Chinese.

The catalogue whisperjav/webview_gui/assets/i18n_zh.js maps each English text the GUI shows to its
Chinese text; I18n in app.js applies it to the window. These tests read every source of a text the
window can show and fail when one has no entry, so a new English text cannot ship untranslated:

  - index.html: text, tooltips, placeholders, aria-labels, menu group labels;
  - app.js: the string literals that reach the window or a pop-up box (log-panel lines and code
    strings are not texts; NOT_UI below lists the code strings the rules cannot tell apart);
  - api.py: the Customize Parameters schemas and the CrispASR status panel lines;
  - the tool YAML files' gui sections and the Hugging Face model list, which api.py forwards.

The console, the file list and messages from the running program stay English (R01; W283 2).
A text that stays as it is (a model or program name, a unit) is listed under "keep".

Also: the catalogue's own form, and get_ui_language / set_ui_language (W283 3).
"""

import ast
import json
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
GUI = REPO / "whisperjav" / "webview_gui"
ASSETS = GUI / "assets"
TOOLS_YAML = REPO / "whisperjav" / "config" / "v4" / "ecosystems" / "tools"
HF_REGISTRY = REPO / "whisperjav" / "config" / "v4" / "ecosystems" / "transformers" / "hf_models_registry.yaml"
ACCELERATORS = REPO / "whisperjav" / "installer" / "core" / "accelerators.py"

PLACEHOLDER = re.compile(r"\{[A-Za-z_]\w*\}")


def norm(s):
    return re.sub(r"[\s\u00a0]+", " ", s).strip()


def has_letters(s):
    return re.search(r"[A-Za-z]{2,}", s) is not None


def skeleton(s):
    """A text with its computed parts as {}: '{tally}', '${x}' and an f-string's {…} all match."""
    return PLACEHOLDER.sub("{}", norm(s))


# ---------------------------------------------------------------- the catalogue

def load_catalogue():
    src = (ASSETS / "i18n_zh.js").read_text(encoding="utf-8")
    m = re.search(r"window\.WJ_I18N_ZH\s*=\s*(\{.*\})\s*;\s*$", src, re.S)
    assert m, "i18n_zh.js must hold `window.WJ_I18N_ZH = {...};` with a JSON object"
    return json.loads(m.group(1))


# ---------------------------------------------------------------- index.html

class _HtmlTexts(HTMLParser):
    """Text nodes and translated attributes, outside the parts I18n never touches."""
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
    SKIP_TAGS = {"script", "style", "code", "textarea", "head"}  # I18n walks the body only

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []  # True where the element is skipped (or inside a skipped one)
        self.found = []  # (where, text)

    def _skipped(self):
        return bool(self.stack) and self.stack[-1]

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        skip = (self._skipped() or tag in self.SKIP_TAGS or "data-i18n-skip" in d
                or "file-path" in (d.get("class") or "").split())
        if not skip:
            line = self.getpos()[0]
            for a in ("title", "placeholder", "aria-label"):
                if d.get(a):
                    self.found.append((f"{tag}@{a} line {line}", d[a]))
            if tag == "optgroup" and d.get("label"):
                self.found.append((f"optgroup@label line {line}", d["label"]))
        if tag not in self.VOID:
            self.stack.append(skip)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in self.VOID:
            self.stack.pop()

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        if self.stack:
            self.stack.pop()

    def handle_data(self, data):
        if not self._skipped() and data.strip():
            self.found.append((f"text line {self.getpos()[0]}", data))


def html_texts(html):
    p = _HtmlTexts()
    p.feed(html)
    return [(where, norm(t)) for where, t in p.found if has_letters(t)]


def index_html_texts():
    return [("index.html " + w, t) for w, t in html_texts((ASSETS / "index.html").read_text(encoding="utf-8"))]


# ---------------------------------------------------------------- app.js

# Calls whose string arguments never reach the window: the log panel and the developer console
# (W283 2: the GUI's log-panel lines stay English), program errors, and code.
LOG_CALLS = {"ConsoleManager.log", "ConsoleManager.appendRaw", "console.log", "console.warn", "console.error",
             "console.info", "console.debug", "ErrorHandler.showWarning", "ErrorHandler.showSuccess",
             "UIHelpers.showSuccess", "Error"}
CODE_CALL_NAMES = {"getElementById", "querySelector", "querySelectorAll", "closest", "matches", "addEventListener",
                   "removeEventListener", "add", "remove", "toggle", "contains", "getItem", "setItem", "removeItem",
                   "createElement", "Event", "CustomEvent", "includes", "startsWith", "endsWith", "split", "join",
                   "indexOf", "getAttribute", "hasAttribute", "removeAttribute", "RegExp", "test", "match",
                   "parse", "stringify", "createTreeWalker", "setTimeout"}
ID_LIKE = re.compile(r"^[\w\-.:/#\[\]=*>+~^$@%(),'\"{}|!?\\]+$")
CSS_CLASSES = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*(?: [a-z0-9]+(?:-[a-z0-9]+)*)+$")

# app.js literals that pass the rules above but are code or console-only, with the reason.
NOT_UI = {
    "Home": "key name", "End": "key name", "Delete": "key name", "Backspace": "key name",
    "Escape": "key name", "ArrowLeft": "key name", "ArrowRight": "key name", "ArrowUp": "key name",
    "ArrowDown": "key name", "Enter": "key name",
    "input, select": "CSS selector",
    "C:\\Users\\Documents\\WhisperJAV\\output": "fallback output folder",
    "--verbose True --standard_asia": "XXL Extra Args value",
    "[data-i18n-skip], .file-path, code, textarea, script, style": "I18n.SKIP selector",
    "console-line {}": "CSS classes",
    "&amp;": "HTML entity", "&lt;": "HTML entity", "&gt;": "HTML entity",
    "Ready.": "console line (ConsoleManager.clear)",
    "SRT file(s)": "console line (Added n SRT file(s))",
    "Files: {}.": "console line (ErrorHandler.showSuccess body)",
    "The run finished.": "console line (ErrorHandler.showSuccess body)",
    "Some files ended empty or suspect (listed below).": "console line (ErrorHandler.showSuccess body)",
    "ollama pull {}": "a command, shown in <code>",
    "Good": "Ollama recommendation data, not shown",
}


def _js_tokens(src, line=1):
    """Tokens of JavaScript source: ('str', text, line, merged_with_next_ok) and ('punct'/'ident', ...).
    Skips comments and regex literals; the ${…} parts of a template are tokenized too ('tpl-inner')."""
    out = []
    i, n = 0, len(src)
    prev = ""  # previous significant token text
    while i < n:
        ch = src[i]
        if ch == "\n":
            line += 1
            i += 1
            continue
        if ch.isspace():
            i += 1
            continue
        if src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j < 0 else j
            continue
        if src.startswith("/*", i):
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            line += src.count("\n", i, j)
            i = j
            continue
        if ch == "/" and (prev in ("", "(", ",", "=", ":", "[", "!", "&", "|", "?", "{", "}", ";", "return",
                                   "&&", "||", "+")):
            j = i + 1
            in_class = False
            while j < n and src[j] != "\n":
                if src[j] == "\\":
                    j += 2
                    continue
                if src[j] == "[":
                    in_class = True
                elif src[j] == "]":
                    in_class = False
                elif src[j] == "/" and not in_class:
                    break
                j += 1
            j += 1
            while j < n and src[j].isalpha():
                j += 1
            out.append(("regex", src[i:j], line))
            prev = "regex"
            i = j
            continue
        if ch in "'\"`":
            q, start, j, buf = ch, line, i + 1, []
            while j < n:
                cj = src[j]
                if cj == "\\":
                    nxt = src[j + 1:j + 2]
                    if nxt == "u" and re.match(r"[0-9A-Fa-f]{4}", src[j + 2:j + 6]):
                        buf.append(chr(int(src[j + 2:j + 6], 16)))
                        j += 6
                        continue
                    buf.append({"n": "\n", "t": "\t"}.get(nxt, nxt if nxt in "'\"`\\" else "\\" + nxt))
                    j += 2
                    continue
                if q == "`" and src.startswith("${", j):
                    depth, k = 1, j + 2
                    while k < n and depth:
                        if src[k] in "'\"`":  # skip a nested literal so its braces do not count
                            qq, k = src[k], k + 1
                            while k < n and src[k] != qq:
                                k += 2 if src[k] == "\\" else 1
                        elif src[k] == "{":
                            depth += 1
                        elif src[k] == "}":
                            depth -= 1
                        k += 1
                    inner = src[j + 2:k - 1]
                    for tok in _js_tokens(inner, line):
                        if tok[0] == "str":
                            out.append(("tpl-inner",) + tok[1:])
                    buf.append("{}")
                    line += src.count("\n", j, k)
                    j = k
                    continue
                if cj == q:
                    break
                if cj == "\n":
                    line += 1
                buf.append(cj)
                j += 1
            out.append(("str", "".join(buf), start))
            prev = "str"
            i = j + 1
            continue
        m = re.match(r"[\w$]+", src[i:])  # names and numbers
        if m:
            out.append(("ident", m.group(0), line))
            prev = m.group(0)
            i += len(m.group(0))
            continue
        m = re.match(r"=>|===|!==|==|!=|&&|\|\||\?\.|\.\.\.|[^\w\s]", src[i:])
        tok = m.group(0)
        out.append(("punct", tok, line))
        prev = tok
        i += len(tok)
    return out


def app_js_literals(src):
    """(line, text, calls) for every string in app.js, '+'-joined strings merged, with the chain of calls
    each sits in (innermost last)."""
    toks = _js_tokens(src)
    result = []
    stack = []    # call name per open '('; '' for a plain parenthesis
    chain = ""    # identifier chain just read: ConsoleManager.log
    pending = None
    k = 0
    while k < len(toks):
        kind, val, line = toks[k]
        if kind == "ident":
            chain = chain + "." + val if (k > 0 and toks[k - 1][1] in (".", "?.") and chain) else val
        elif kind == "punct":
            if val == "(":
                stack.append(chain)
                chain = ""
            elif val == ")":
                if stack:
                    stack.pop()
                chain = ""
            elif val not in (".", "?."):
                chain = ""
        if kind in ("str", "tpl-inner"):
            text = val
            # merge  'a' + 'b'  (+ 'c' ...)
            while (kind == "str" and k + 2 < len(toks) and toks[k + 1][1] == "+"
                   and toks[k + 2][0] == "str"):
                text += toks[k + 2][1]
                k += 2
            result.append((line, text, list(stack)))
            chain = ""
        k += 1
    return result


def _js_is_code(text):
    t = norm(text)
    if not has_letters(t):
        return True
    if " " not in t and ID_LIKE.match(t) and not re.match(r"^[A-Z][a-z]+(?: ?[A-Z0-9][a-z0-9]*)*[.:!?]*$", t):
        return True
    if CSS_CLASSES.match(t) and "-" in t:
        return True
    return False


def app_js_texts():
    src = (ASSETS / "app.js").read_text(encoding="utf-8")
    out = []
    for line, text, calls in app_js_literals(src):
        if any(c in LOG_CALLS or c.startswith("pywebview.api") for c in calls):
            continue
        if calls and calls[-1].split(".")[-1] in CODE_CALL_NAMES:
            continue
        if "<" in text and re.search(r"</?[a-z][a-z0-9]*[\s>/]", text):
            for where, frag in html_texts(text):
                if frag not in NOT_UI:
                    out.append((f"app.js line {line} (HTML {where})", frag))
            continue
        if _js_is_code(text) or norm(text) in NOT_UI:
            continue
        out.append((f"app.js line {line}", norm(text)))
    return out


# ---------------------------------------------------------------- api.py and accelerators.py

# The functions that build what the Customize Parameters window shows, and the keys of their texts.
SCHEMA_FUNCS = {"_get_qwen_schema_base", "get_qwen_schema", "get_enhancer_schema", "get_transformers_schema",
                "get_cohere_schema", "_aligner_and_timing_schema", "_native_vad_schema", "get_segmenter_schema",
                "get_scene_detector_schema", "_load_hf_models_registry"}
SCHEMA_KEYS = {"label", "description", "info_message", "display_name", "warning", "placeholder", "hint"}


def _py_str(node):
    """A string expression as text, an f-string's computed parts as {}; None for anything else."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(str(v.value) if isinstance(v, ast.Constant) else "{}" for v in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        a, b = _py_str(node.left), _py_str(node.right)
        return a + b if a is not None and b is not None else None
    return None


def api_py_texts():
    tree = ast.parse((GUI / "api.py").read_text(encoding="utf-8"))
    out = []

    def walk(node, func, cls_attr):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func = node.name
        if isinstance(node, ast.Dict) and (func in SCHEMA_FUNCS or cls_attr == "_NATIVE_VAD_GUI"):
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and k.value in SCHEMA_KEYS:
                    s = _py_str(v)
                    if s and has_letters(s):
                        out.append((f"api.py line {v.lineno} ({func or cls_attr}.{k.value})", norm(s)))
        # an option label assembled in a variable (the Alignment tab's aligner models)
        if func in SCHEMA_FUNCS and isinstance(node, (ast.Assign, ast.AugAssign)):
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            if isinstance(target, ast.Name) and target.id == "label":
                s = _py_str(node.value)
                if s and has_letters(s):
                    s = "{}" + s if isinstance(node, ast.AugAssign) else s
                    out.append((f"api.py line {node.lineno} ({func}.label)", norm(s)))
        for ch in ast.iter_child_nodes(node):
            walk(ch, func, cls_attr)

    for top in tree.body:
        if isinstance(top, ast.ClassDef):
            for item in top.body:
                if isinstance(item, ast.Assign) and len(item.targets) == 1 and isinstance(item.targets[0], ast.Name):
                    name = item.targets[0].id
                    # The CrispASR status panel (route B design 12, T16; item 12)
                    if name.startswith("CRISPASR_") and isinstance(item.value, ast.Constant) \
                            and isinstance(item.value.value, str):
                        out.append((f"api.py line {item.lineno} ({name})", norm(item.value.value)))
                    walk(item.value, None, name)
                else:
                    walk(item, None, None)
        else:
            walk(top, None, None)

    acc = ast.parse(ACCELERATORS.read_text(encoding="utf-8"))
    for item in acc.body:
        if isinstance(item, ast.Assign) and isinstance(item.targets[0], ast.Name) \
                and item.targets[0].id.startswith("GUI_CRISPASR_RUNS_ON"):
            values = item.value.values if isinstance(item.value, ast.Dict) else [item.value]
            for v in values:
                s = _py_str(v)
                if s:
                    out.append((f"accelerators.py line {v.lineno} ({item.targets[0].id})", norm(s)))
    return out


def yaml_texts():
    out = []
    for yf in sorted(TOOLS_YAML.glob("*.yaml")):
        data = yaml.safe_load(yf.read_text(encoding="utf-8")) or {}
        meta = data.get("metadata") or {}
        for k in ("displayName", "description"):
            if isinstance(meta.get(k), str) and has_letters(meta[k]):
                out.append((f"{yf.name} metadata.{k}", norm(meta[k])))
        for name, hints in (data.get("gui") or {}).items():
            if not isinstance(hints, dict):
                continue
            # api.py _convert_gui_hints_to_schema: a missing label is the parameter name, title-cased
            out.append((f"{yf.name} gui.{name}.label", norm(hints.get("label", name.replace("_", " ").title()))))
            if isinstance(hints.get("description"), str) and has_letters(hints["description"]):
                out.append((f"{yf.name} gui.{name}.description", norm(hints["description"])))
            for opt in hints.get("options") or []:
                if isinstance(opt, dict) and isinstance(opt.get("label"), str):
                    out.append((f"{yf.name} gui.{name}.options", norm(opt["label"])))
    reg = yaml.safe_load(HF_REGISTRY.read_text(encoding="utf-8")) or {}
    for m in reg.get("models") or []:
        if isinstance(m, dict) and isinstance(m.get("label"), str):
            out.append((f"{HF_REGISTRY.name} models.label", norm(m["label"])))
    return out


def js_title(name):
    """app.js's label for a name: name.replace(/_/g, ' ').replace(/\\b\\w/g, upper)."""
    return re.sub(r"\b\w", lambda m: m.group(0).upper(), name.replace("_", " "))


def title_cased_texts():
    """Labels app.js builds from internal names: the Quality tab's parameters (generateQualityTab's
    params lists) and the group headings of the Segmenter, Enhancer and Scene tabs (the 'group' of each
    YAML gui entry and of api.py's schemas)."""
    out = []
    src = (ASSETS / "app.js").read_text(encoding="utf-8")
    start = src.index("async generateQualityTab(")
    block = src[start:src.index("for (const [groupKey", start)]
    for lst in re.findall(r"params:\s*\[([^\]]*)\]", block):
        for name in re.findall(r"'([a-z_0-9]+)'", lst):
            out.append(("app.js generateQualityTab params", js_title(name)))
    for yf in sorted(TOOLS_YAML.glob("*.yaml")):
        data = yaml.safe_load(yf.read_text(encoding="utf-8")) or {}
        for name, hints in (data.get("gui") or {}).items():
            if isinstance(hints, dict):
                out.append((f"{yf.name} gui.{name}.group", js_title(str(hints.get("group", "general")))))
    for node in ast.walk(ast.parse((GUI / "api.py").read_text(encoding="utf-8"))):
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and k.value == "group" and isinstance(v, ast.Constant) \
                        and isinstance(v.value, str):
                    out.append((f"api.py line {v.lineno} (group)", js_title(v.value)))
    return out


def all_shown_texts():
    return index_html_texts() + app_js_texts() + api_py_texts() + yaml_texts() + title_cased_texts()


def uncovered(texts, catalogue):
    covered = {skeleton(k) for k in catalogue["texts"]} | {skeleton(k) for k in catalogue["keep"]}
    missing = {}
    for where, text in texts:
        if skeleton(text) not in covered:
            missing.setdefault(text, where)
    return missing


# ---------------------------------------------------------------- tests: coverage

@pytest.fixture(scope="module")
def catalogue():
    return load_catalogue()


@pytest.mark.parametrize("source", ["index.html", "app.js", "api.py", "yaml", "title-cased names"])
def test_every_shown_text_has_chinese_or_is_kept(source, catalogue):
    texts = {"index.html": index_html_texts, "app.js": app_js_texts, "api.py": api_py_texts,
             "yaml": yaml_texts, "title-cased names": title_cased_texts}[source]()
    assert texts, f"no texts found in {source}: the extraction is broken"
    missing = uncovered(texts, catalogue)
    assert not missing, (f"{len(missing)} {source} texts have no entry in i18n_zh.js (add the Chinese under "
                         "'texts', or the text under 'keep' when it stays as it is):\n"
                         + "\n".join(f"  {w}: {t!r}" for t, w in sorted(missing.items(), key=lambda x: x[1])))


def test_extraction_finds_known_texts():
    """The extractors themselves: one known text from each kind of place."""
    html = {t for _, t in index_html_texts()}
    assert {"Add File(s)", "Remember settings", "Pass 1", "Whisper-Based", "Output directory..."} <= html
    assert "Ready." not in html  # inside the console (data-i18n-skip)
    js = {t for _, t in app_js_texts()}
    assert "Finished · {}" in js                        # a template literal
    assert "No configurable parameters for this segmenter." in js   # text inside an HTML literal
    assert not any(t.startswith("Process started successfully") for t in js)  # a log-panel line
    api = {t for _, t in api_py_texts()}
    assert "Speech Threshold" in api and "Model: {label} — downloaded" in api
    assert "GPU: asks for the NVIDIA GPU (CUDA)" in api
    assert "Process already running" not in api         # a program message, stays English
    assert "Silero VAD" in {t for _, t in yaml_texts()}


# ---------------------------------------------------------------- tests: the catalogue's form

def test_catalogue_form(catalogue):
    assert set(catalogue) == {"texts", "keep", "contexts"}
    problems = []
    pairs = list(catalogue["texts"].items())
    for selector, texts in catalogue["contexts"].items():
        # a context entry translates, inside its selector, a text kept or translated elsewhere
        pairs += list(texts.items())
        for en in texts:
            if en not in catalogue["texts"] and en not in catalogue["keep"]:
                problems.append(f"context {selector!r}: {en!r} is neither in texts nor in keep")
    for en, zh in pairs:
        if en != norm(en):
            problems.append(f"key not whitespace-normalized: {en!r}")
        if not isinstance(zh, str) or not zh.strip():
            problems.append(f"empty Chinese for {en!r}")
        elif sorted(PLACEHOLDER.findall(en)) != sorted(PLACEHOLDER.findall(zh)):
            problems.append(f"placeholders differ: {en!r} -> {zh!r}")
    both = set(catalogue["texts"]) & set(catalogue["keep"])
    problems += [f"in texts and keep: {t!r}" for t in sorted(both)]
    assert not problems, "\n".join(problems)


def test_owner_word_rules(catalogue):
    """W10: no 识别 ("recognition"); W286-W287 replaced 通道 (pass), 激进/保守 (sensitivity), 转录 (transcription)
    and 集成模式 (the Ensemble tab, now 专家模式)."""
    values = list(catalogue["texts"].values())
    for texts in catalogue["contexts"].values():
        values += list(texts.values())
    bad = [(w, v) for v in values for w in ("识别", "通道", "激进", "保守", "转录", "集成模式") if w in v]
    assert not bad, bad
    assert catalogue["texts"]["Ensemble Mode"] == "专家模式" and catalogue["texts"]["Transcription Mode"] == "基础模式"


def test_reverse_lookup_is_unambiguous(catalogue):
    """Switching back to English reads the catalogue backwards for copied nodes; one Chinese text
    must name one English text (I18n._reverse)."""
    seen, clashes = {}, []
    pairs = list(catalogue["texts"].items())
    for texts in catalogue["contexts"].values():
        pairs += list(texts.items())
    for en, zh in pairs:
        if PLACEHOLDER.search(en):
            continue
        z = norm(zh)
        # W287: the AI SRT Translate tab and the Expert Mode tick box are both AI 字幕翻译, by the owner's choice
        if z == "AI 字幕翻译":
            continue
        if z in seen and seen[z] != en:
            clashes.append(f"{z!r}: {seen[z]!r} and {en!r}")
        seen[z] = en
    assert not clashes, "\n".join(clashes)


# ---------------------------------------------------------------- tests: the saved language (W283 3)

@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr("whisperjav.settings.gui_settings.get_gui_settings_path",
                        lambda: tmp_path / "gui_settings.json")
    from whisperjav.webview_gui.api import WhisperJAVAPI
    return WhisperJAVAPI()


@pytest.mark.parametrize("chinese, expected", [(True, "zh-CN"), (False, "en")])
def test_first_start_follows_display_language(api, monkeypatch, chinese, expected):
    monkeypatch.setattr(type(api), "_display_language_is_chinese", staticmethod(lambda: chinese))
    assert api.get_ui_language() == {"success": True, "language": expected, "saved": False}


def test_choice_is_kept_with_remember_settings_off(api, monkeypatch, tmp_path):
    monkeypatch.setattr(type(api), "_display_language_is_chinese", staticmethod(lambda: False))
    assert api.set_ui_language("zh-CN") == {"success": True}
    # the form saves with Remember settings off: only rememberSettings is sent
    assert api.save_gui_settings({"rememberSettings": False})["success"]
    assert api.get_ui_language() == {"success": True, "language": "zh-CN", "saved": True}
    on_disk = json.loads((tmp_path / "gui_settings.json").read_text(encoding="utf-8"))
    assert on_disk["ui_language"] == "zh-CN" and on_disk["remember_settings"] is False
    assert api.set_ui_language("en") == {"success": True}
    assert api.get_ui_language()["language"] == "en"


def test_unknown_language_is_refused(api, tmp_path):
    assert api.set_ui_language("fr")["success"] is False
    assert not (tmp_path / "gui_settings.json").exists()


def test_display_language_check_never_raises(api):
    assert api._display_language_is_chinese() in (True, False)
