# 📄 DSearch – Vector-based document search that speaks CLI & Claude  

DSearch is a tiny two-piece toolkit  

1. `main.py` – a **Python-powered HTTP server** that keeps a local
   [Chroma](https://github.com/chroma-core/chroma) vector database of your text files.  
2. `dsearch` – a **Bash wrapper** that turns the REST API into a familiar
   command-line interface and prints results in the plain-text format preferred by
   the Claude **code** / “Code Agent”.

Put both together and you have an “index → search → get answer” workflow that can
be scripted, automated or called from inside Claude without writing a single
line of Python.

---

## ✨ Features

•   Recursive indexing of any directory (code, Markdown, JSON, …)  
•   Add standalone snippets on the fly – they are stored as real files, too  
•   Cosine similarity search with tunable threshold & max-results  
•   Automatic sync: on startup the DB is reconciled with the file-system  
•   Pure standard library + Chroma – no heavy web framework needed  
•   One-file CLI (`dsearch`) – easy to drop into other projects / agents  
•   Output layout designed so Claude can “just read it”

---

## 🚀 Quick start

```bash
git clone https://github.com/you/dsearch.git
cd dsearch

# 0)  Install Python deps
python3 -m pip install chromadb

# 1)  Make the wrapper executable
chmod +x dsearch

# mv dsearch to somewhere in your system path!

# 2)  Start the vector-search server
python3 main.py server --host 0.0.0.0 --port 8080 &
# (leave it running in the background or a tmux pane)

# 4)  Fire a query
dsearch -s "distributed tracing"

# 5)  Add an ad-hoc snippet
dsearch -a "ChromaDB is awesome" --title "note"
```

---

## 🗄️ Repository layout

```
.
├── main.py   # the HTTP/Chroma engine
├── dsearch.sh           # CLI wrapper (curl + JSON helpers)
└── README.md            # you are here
```

---

## 🔧 Configuration

Both parts are be tuned independently.

### Server (`main.py`)

```bash
python3 main.py server \
    --host 0.0.0.0 \
    --port 8080 \
    --db-path ./db \
    --collection documents \
    --index-path ./documents \
    --max-results 8 \
    --similarity-threshold 0.6
```

During startup the server scans `--index-path` and makes sure every file is
represented exactly once in the vector store (deleted files are pruned,
changed files are re-indexed).

Other sub-commands (run without the Bash wrapper):

```
python3 main.py index  <directory>   # one-shot indexing
python3 main.py search "<query>"     # quick test search
python3 main.py add    "<content>"   # add snippet
python3 main.py test                 # self-test
```

### Client (`dsearch`)

Environment variables override defaults so you don’t have to repeat the flags:

```
export DOCSEARCH_HOST=localhost
export DOCSEARCH_PORT=8080
```

Full help:

```
dsearch --help
```

```
Actions
  -i, --index <directory>      Recursively index directory
  -s, --search <query>         Search the index
  -a, --add <string>           Add single text snippet
  -t, --test                   Run server self-test

Options
  --max-results <n>            Max hits     (default 5)
  --similarity-threshold <t>   Similarity   (default 0.7)
  --index-path <path>          Remote path used by --add
  --title <text>               Optional title when adding
  --host / --port              Server location
```

---

## Automatic setup
Copy the .claude/agents folder into your project folder where you execute the claude command from.

## 🤖 Manual setup from Claude

Because the wrapper prints **plain text** (no ANSI, no JSON unless asked),
Claude’s code interpreter can consume it straight away:

First create your agent
```claude```

type:
```/agents```

Create a new agent:
```
Create new agent -> project -> Manual -> enter any text -> enter any text -> enter any text
```

Customize tools:
```
The agent should ONLY have access to execution tasks.
De-select all-tools. Enable execution only!
Show advanced options -> Enable bash
```

Need machine-readable output?  Append `&format=json` to any `/search` request or
just call the server REST endpoint directly.

---

## 🛠️ Development

* Python ≥ 3.9  
* Chroma ≥ 0.4.0  

Optional but handy:

```bash
python3 -m pip install black isort ruff
```

Run unit-style smoke test:

```bash
python3 main.py test
```

---

## 📜 License

MIT – do whatever you want but give credit. Pull requests welcome!