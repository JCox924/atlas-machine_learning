# QA Bot Project

This project implements a Question-Answering (QA) chatbot using:  
- A pre-trained BERT model for extractive QA (TensorFlow Hub)  
- The Hugging Face transformers library for tokenization  
- Universal Sentence Encoder (TensorFlow Hub) for semantic search

## Learning Objectives

- **General Concepts**  
  - Explain what Question-Answering is and how extractive QA works.  
  - Describe Semantic Search and its applications.  
  - Understand the BERT architecture and its fine‑tuned QA variants.  
  - Build a QA chatbot pipeline from retrieval to answer extraction.  
  - Use the `transformers` library for tokenization.  
  - Use the `tensorflow-hub` library for model loading (QA & text embeddings).

## Project Structure

```
supervised_learning/qa_bot/
├── README.md
├── ZendeskArticles/         # Provided reference documents (.md files)
├── 0-qa.py                  # Single-document extractive QA function
├── 1-loop.py                # Interactive prompt scaffold (empty answers)
├── 2-qa.py                  # Answer loop over one reference text
├── 3-semantic_search.py     # Semantic search over the article corpus
└── 4-qa.py                  # Multi-document QA loop using retrieval + QA
```

## Requirements

- **Editor**: `vi`, `vim`, or `emacs`  
- **OS**: Ubuntu 20.04 LTS  
- **Python**: 3.9.x  
- **Packages**:  
  - `numpy==1.25.2`  
  - `tensorflow==2.15`  
  - `tensorflow-hub==0.15.0`  
  - `transformers==4.44.2`

All code files must:

- Begin with the shebang: `#!/usr/bin/env python3`  
- End with a newline  
- Be executable (`chmod +x <filename>.py`)  
- Pass `pycodestyle` (2.11.1) checks  
- Include module/class/function docstrings accessible via Python introspection

## Installation

```bash
# Create a virtual environment (optional)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --user numpy==1.25.2 tensorflow==2.15 \
            tensorflow-hub==0.15.0 transformers==4.44.2
```

## Usage

### 1) Single-Document QA (`0-qa.py`)

```bash
chmod +x 0-qa.py
./0-qa.py  # (no interactive behavior)  
# Use from Python:
python3 - << 'EOF'
from 0-qa import question_answer
ref = open('ZendeskArticles/PeerLearningDays.md').read()
print(question_answer('When are PLDs?', ref))
EOF
```

### 2) Interactive Loop Scaffold (`1-loop.py`)

Prompts `Q: ` and prints `A: ` (empty). Exits on `exit`, `quit`, `goodbye`, or `bye`.

```bash
chmod +x 1-loop.py
./1-loop.py
```

### 3) Answer Loop with QA (`2-qa.py`)

Uses the extractive QA model over a single reference text.

```bash
chmod +x 2-qa.py
# Example:
python3 - << 'EOF'
from 2-qa import answer_loop
ref = open('ZendeskArticles/PeerLearningDays.md').read()
answer_loop(ref)
EOF
```

### 4) Semantic Search (`3-semantic_search.py`)

Retrieves the document most similar to a query using USE embeddings.

```bash
chmod +x 3-semantic_search.py
./3-semantic_search.py  # no direct CLI; use from Python:
python3 - << 'EOF'
from 3-semantic_search import semantic_search
print(semantic_search('ZendeskArticles', 'When are PLDs?'))
EOF
```

### 5) Multi-Reference QA Loop (`4-qa.py`)

Combines semantic search + extractive QA into an interactive chatbot.

```bash
chmod +x 4-qa.py
./4-qa.py   # enter questions at prompts
```

## Notes

- Ensure the `ZendeskArticles/` directory remains in the project root.  
- All interactive scripts handle graceful exit on EOF or recognized exit commands.  
- For any QA failures (no span found), the bot responds with a polite fallback.

---
