"""Step 2: Streamlit app for reviewing and correcting the Manado side of the labelled data.

Shows each Formal sentence next to its Manado translation, lets you edit the
Manado text, and has a bulk find-and-replace for standardising spellings.
Changes are saved straight back to the file.

Usage:
    streamlit run tools/dataset_editor.py
    streamlit run tools/dataset_editor.py -- --file path/to/other.jsonl

Shortcut: press backslash (\\) to save and go to the next pair.
"""

import argparse
import json
import re
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

DEFAULT_FILE = Path(__file__).resolve().parent.parent / "data" / "labeled.jsonl"
DIALECT_KEY = "manado"
OFFICIAL_KEY = "bahasa"

NEXT_SHORTCUT_JS = """
<script>
const doc = window.parent.document;
if (!doc.getElementById('shortcut-listener')) {
    const marker = doc.createElement('div');
    marker.id = 'shortcut-listener';
    doc.body.appendChild(marker);
    doc.addEventListener('keydown', function (e) {
        if (e.key === '\\\\') {
            e.preventDefault();
            const next = Array.from(doc.querySelectorAll('button')).find(el => el.innerText.includes('Next'));
            if (next) next.click();
        }
    });
}
</script>
"""


def get_file_path() -> Path:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE)
    args, _ = parser.parse_known_args()
    return args.file


def load_data(path: Path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_data(path: Path, data):
    with path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def bulk_replace(path: Path, dialect_indices):
    with st.expander("🛠️ Bulk Find & Replace (Dialect Data)"):
        st.markdown("Replace several spellings of a word with one standard spelling across the **entire dataset**.")

        col_target, col_vars = st.columns([1, 2])
        target_word = col_target.text_input("Replace with (Target):", placeholder="e.g., Kiapa")
        variations = col_vars.text_input("Words to replace (comma-separated):", placeholder="e.g., kyapa, kyp, kpa")

        col1, col2 = st.columns(2)
        whole_word = col1.checkbox("Match whole word only", value=True, help="Prevents replacing 'kpa' inside 'pakpakpa'")
        match_case = col2.checkbox("Match case", value=False)

        if not st.button("Replace All", type="primary"):
            return
        var_list = [v.strip() for v in variations.split(",") if v.strip()]
        if not target_word or not var_list:
            st.warning("Please provide both a target word and at least one variation.")
            return

        pattern_str = "|".join(re.escape(v) for v in var_list)
        if whole_word:
            pattern_str = rf"\b({pattern_str})\b"
        pattern = re.compile(pattern_str, 0 if match_case else re.IGNORECASE)

        data = st.session_state.data
        replaced = modified = 0
        for i in dialect_indices:
            new_text, count = pattern.subn(target_word, data[i].get("input", ""))
            if count:
                data[i]["input"] = new_text
                replaced += count
                modified += 1

        if modified:
            save_data(path, data)
            st.success(f"Replaced {replaced} instance(s) across {modified} translation(s)!")
        else:
            st.info("No matches found to replace.")


def main():
    path = get_file_path()
    st.set_page_config(page_title="Dataset Editor", layout="centered")
    st.title("JSONL Translation Editor")
    st.caption(str(path))

    if "data" not in st.session_state:
        st.session_state.data = load_data(path)
    data = st.session_state.data

    if not data:
        st.error(f"Could not find or load {path}.")
        st.stop()

    dialect_indices = [i for i, item in enumerate(data) if item.get("output") == DIALECT_KEY]
    if not dialect_indices:
        st.warning("No dialect entries found in the dataset.")
        st.stop()

    bulk_replace(path, dialect_indices)
    st.markdown("---")

    # Navigation
    if "current_idx_pos" not in st.session_state:
        st.session_state.current_idx_pos = 0
    total = len(dialect_indices)
    pos = st.session_state.current_idx_pos
    idx = dialect_indices[pos]

    st.sidebar.markdown("### Navigation")
    jump_to = st.sidebar.slider("Jump to pair:", 1, total, pos + 1)
    if jump_to - 1 != pos:
        st.session_state.current_idx_pos = jump_to - 1
        st.rerun()
    st.sidebar.progress((pos + 1) / total)
    st.sidebar.text(f"Pair {pos + 1} of {total}")

    # The Formal sentence is the line just before its Manado translation.
    official_text = "N/A (No preceding official translation found)"
    if idx > 0 and data[idx - 1].get("output") == OFFICIAL_KEY:
        official_text = data[idx - 1].get("input")

    st.subheader("Official Language:")
    st.info(official_text)
    st.subheader("Dialect (Editable):")
    new_text = st.text_area("Edit text here:", value=data[idx].get("input", ""), height=100)

    col1, col2, col3 = st.columns(3)
    if col1.button("⬅️ Previous", use_container_width=True) and pos > 0:
        st.session_state.current_idx_pos -= 1
        st.rerun()
    if col2.button("💾 Save Changes", use_container_width=True):
        data[idx]["input"] = new_text
        save_data(path, data)
        st.success("Saved directly to file!")
    if col3.button("Next ➡️", use_container_width=True) and pos < total - 1:
        data[idx]["input"] = new_text
        save_data(path, data)
        st.session_state.current_idx_pos += 1
        st.rerun()

    components.html(NEXT_SHORTCUT_JS, height=0, width=0)


main()
