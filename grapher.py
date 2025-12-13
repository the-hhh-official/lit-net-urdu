import re
import json
from glob import glob
from collections import defaultdict
from docx import Document

# ========== CONFIG ==========
name = "ZaatEShareef"

DOCX_GLOB = "C:/Users/LENOVO/OneDrive/Desktop/SNA Proj/SNA/JSONS,CSVs and Graphing/Temp/"+name+"-*_results.docx"
ALIAS_JSON_PATH = "C:/Users/LENOVO/OneDrive/Desktop/SNA Proj/SNA/JSONS,CSVs and Graphing/Temp/ZaatEShareef_aliases.json"

NODES_FILE = name+"_nodes.txt"
EDGES_FILE_TEMPLATE = name+"_edges_w{w}.csv"
ANALYSIS_FILE = name+"_graph_analysis.txt"

ALWAYS_INCLUDE_QAYUM = False
QAYUM_NAME = "قیوم"

# We will run for all these windows:
WINDOWS = [0, 5, 10]


# ========== HELPERS ==========

def load_alias_map(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    alias_map_active = {}
    active_set = set()

    for canonical, val in data.items():
        if isinstance(val, list):
            aliases = [a for a in val if a and isinstance(a, str)]
            active = True
        elif isinstance(val, dict):
            aliases = [a for a in val.get("aliases", []) if a and isinstance(a, str)]
            active = bool(val.get("active", True))
        else:
            continue

        if canonical not in aliases:
            aliases.append(canonical)

        if active:
            alias_map_active[canonical] = sorted(
                set(a.strip() for a in aliases),
                key=len,
                reverse=True
            )
            active_set.add(canonical)

    return alias_map_active, active_set


def load_docx_text(path):
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def load_all_text(glob_pattern):
    files = sorted(glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No files match pattern: {glob_pattern}")
    return "\n".join(load_docx_text(f) for f in files)


def split_into_pages(full_text):
    parts = re.split(r'<<<(\d+)>>>', full_text)
    pages = {}
    for i in range(1, len(parts), 2):
        try:
            page_num = int(parts[i])
        except ValueError:
            continue
        pages[page_num] = parts[i + 1]
    if not pages:
        raise ValueError("No <<<n>>> page markers found in text.")
    return pages


def build_alias_patterns(alias_map):
    entries = []
    boundary = r"0-9A-Za-z\u0600-\u06FF"
    for canonical, aliases in alias_map.items():
        for alias in aliases:
            pattern = rf'(?<![{boundary}]){re.escape(alias)}(?![{boundary}])'
            entries.append((alias, re.compile(pattern), canonical))
    entries.sort(key=lambda x: len(x[0]), reverse=True)
    return [(regex, canonical) for (_, regex, canonical) in entries]


def detect_characters_on_page(page_text, alias_patterns, active_set):
    found = set()
    for regex, canonical in alias_patterns:
        if canonical in found:
            continue
        if regex.search(page_text):
            found.add(canonical)

    if ALWAYS_INCLUDE_QAYUM and (QAYUM_NAME in active_set):
        found.add(QAYUM_NAME)

    return found


def build_occurrence_index(pages, alias_patterns, active_set):
    """
    Returns:
      pages_to_chars: dict[int] -> set[str] (who appears on each page)
      chars_to_pages: dict[str] -> sorted list[int] (where each character appears)
    """
    pages_to_chars = {}
    chars_to_pages = defaultdict(set)

    for page_num in sorted(pages.keys()):
        text = pages[page_num]
        chars = detect_characters_on_page(text, alias_patterns, active_set)
        if not chars:
            continue
        pages_to_chars[page_num] = chars
        for c in chars:
            chars_to_pages[c].add(page_num)

    # normalize to sorted lists
    chars_to_pages = {c: sorted(ps) for c, ps in chars_to_pages.items()}
    return pages_to_chars, chars_to_pages


def cooccurrence_matches(pages_a, pages_b, window):
    """
    Two-pointer sweep to find 'matches' where |pa - pb| <= window.
    Returns a list of (pa, pb) pairs (deduplicated, stable) and weight=len(pairs).
    """
    i = j = 0
    matches = []
    while i < len(pages_a) and j < len(pages_b):
        pa, pb = pages_a[i], pages_b[j]
        diff = pa - pb
        if abs(diff) <= window:
            matches.append((pa, pb))
            # advance the one with the earlier page to find new matches
            if pa <= pb:
                i += 1
            else:
                j += 1
        elif diff < 0:
            i += 1
        else:
            j += 1
    return matches


def build_graph(pages, alias_patterns, active_set, page_window):
    """
    If page_window == 0:
        Edge if two characters appear on the EXACT same page.
        Weight = number of pages they share.
        Pages column lists those page numbers.
    If page_window > 0:
        Edge if two characters appear within +/- page_window pages.
        Weight = number of matched (pa, pb) pairs via two-pointer sweep.
        Pages column lists the representative page 'centers' (rounded average of pa, pb).
        Pairs column lists the exact pa:pb pairs.
    """
    pages_to_chars, chars_to_pages = build_occurrence_index(pages, alias_patterns, active_set)

    nodes = set(chars_to_pages.keys())
    edges = defaultdict(lambda: {"pages": set(), "pairs": []})

    chars = sorted(nodes)
    for a_idx in range(len(chars)):
        for b_idx in range(a_idx + 1, len(chars)):
            a, b = chars[a_idx], chars[b_idx]

            if page_window == 0:
                # same-page co-occurrence
                shared_pages = set(chars_to_pages[a]).intersection(chars_to_pages[b])
                if shared_pages:
                    edges[(a, b)]["pages"].update(shared_pages)
            else:
                # windowed co-occurrence
                pairs = cooccurrence_matches(chars_to_pages[a], chars_to_pages[b], page_window)
                if pairs:
                    edges[(a, b)]["pairs"].extend(pairs)
                    # store a representative "center" page (integer average) for readability
                    centers = {(pa + pb) // 2 for (pa, pb) in pairs}
                    edges[(a, b)]["pages"].update(centers)

    return nodes, edges


def write_nodes(nodes, path):
    with open(path, "w", encoding="utf-8") as f:
        for n in sorted(nodes):
            f.write(n + "\n")


def write_edges(edges, path, page_window):
    with open(path, "w", encoding="utf-8") as f:
        if page_window == 0:
            f.write("Character_A,Character_B,Weight,Pages\n")
            for (a, b), data in sorted(edges.items()):
                pages_sorted = sorted(data["pages"])
                if not pages_sorted:
                    continue
                weight = len(pages_sorted)
                pages_str = " ".join(str(p) for p in pages_sorted)
                f.write(f"\"{a}\",\"{b}\",{weight},\"{pages_str}\"\n")
        else:
            # Include both a readable center 'Pages' and exact 'Pairs'
            f.write("Character_A,Character_B,Weight,Pages(centers),Pairs(pa:pb)\n")
            for (a, b), data in sorted(edges.items()):
                pairs = data["pairs"]
                if not pairs:
                    continue
                weight = len(pairs)
                centers = sorted(data["pages"])
                centers_str = " ".join(str(c) for c in centers)
                pairs_str = " ".join(f"{pa}:{pb}" for (pa, pb) in pairs)
                f.write(f"\"{a}\",\"{b}\",{weight},\"{centers_str}\",\"{pairs_str}\"\n")


# ======== ANALYSIS HELPERS =========

def compute_graph_stats(nodes, edges):
    """
    Compute basic graph stats:
      - num_nodes
      - num_edges
      - degrees per node
      - max degree and which nodes have it
      - average degree
      - number of isolated nodes (degree 0)
    """
    num_nodes = len(nodes)
    num_edges = len(edges)

    degrees = {n: 0 for n in nodes}
    for (a, b), data in edges.items():
        # Edges exist only when they have pages/pairs, so we just count them
        degrees[a] += 1
        degrees[b] += 1

    if degrees:
        max_degree = max(degrees.values())
        max_degree_nodes = sorted(n for n, d in degrees.items() if d == max_degree)
        isolated_nodes = sum(1 for d in degrees.values() if d == 0)
        avg_degree = (2 * num_edges / num_nodes) if num_nodes > 0 else 0.0
    else:
        max_degree = 0
        max_degree_nodes = []
        isolated_nodes = num_nodes
        avg_degree = 0.0

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "degrees": degrees,
        "max_degree": max_degree,
        "max_degree_nodes": max_degree_nodes,
        "isolated_nodes": isolated_nodes,
        "avg_degree": avg_degree,
    }


def write_analysis(analysis_results, path):
    """
    analysis_results: list of dicts, each with:
      - window
      - edges_file
      - stats (dict from compute_graph_stats)
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("Graph analysis summary for different PAGE_WINDOW values\n")
        f.write("======================================================\n\n")

        for result in sorted(analysis_results, key=lambda r: r["window"]):
            w = result["window"]
            edges_file = result["edges_file"]
            stats = result["stats"]

            f.write(f"PAGE_WINDOW = {w}\n")
            f.write(f"Edges CSV   = {edges_file}\n")
            f.write(f"Number of nodes       : {stats['num_nodes']}\n")
            f.write(f"Number of edges       : {stats['num_edges']}\n")
            f.write(f"Average degree        : {stats['avg_degree']:.3f}\n")
            f.write(f"Max degree            : {stats['max_degree']}\n")
            f.write("Nodes with max degree : ")
            if stats["max_degree_nodes"]:
                f.write(", ".join(stats["max_degree_nodes"]) + "\n")
            else:
                f.write("(none)\n")
            f.write(f"Isolated nodes (deg=0): {stats['isolated_nodes']}\n")
            f.write("\n")

        f.write("Note: Nodes are the same across all PAGE_WINDOW runs, but edges\n")
        f.write("      become denser as PAGE_WINDOW increases.\n")


# ========== MAIN ==========

def main():
    alias_map_active, active_set = load_alias_map(ALIAS_JSON_PATH)
    full_text = load_all_text(DOCX_GLOB)
    pages = split_into_pages(full_text)
    alias_patterns = build_alias_patterns(alias_map_active)

    analysis_results = []
    nodes_written = False

    for window in WINDOWS:
        print(f"Building graph for PAGE_WINDOW = {window} ...")
        nodes, edges = build_graph(pages, alias_patterns, active_set, window)

        # Write nodes once (they don't depend on window size)
        if not nodes_written:
            write_nodes(nodes, NODES_FILE)
            nodes_written = True
            print(f"Wrote {len(nodes)} nodes to {NODES_FILE}")

        edges_file = EDGES_FILE_TEMPLATE.format(w=window)
        write_edges(edges, edges_file, window)

        kept_edges = len(edges)
        print(f"Wrote {kept_edges} edges to {edges_file} (window={window})")

        # Collect stats for analysis
        stats = compute_graph_stats(nodes, edges)
        analysis_results.append({
            "window": window,
            "edges_file": edges_file,
            "stats": stats,
        })

    # Write combined analysis
    write_analysis(analysis_results, ANALYSIS_FILE)
    print(f"Wrote combined analysis to {ANALYSIS_FILE}")


if __name__ == "__main__":
    main()