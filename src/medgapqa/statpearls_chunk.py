"""StatPearls (NCBI Bookshelf NBK430685) .nxml → paragraph-level snippet JSONL.

The chunking rules port the logic of MedRAG (Xiong et al., 2024, https://github.com/Teddy-XiongGZ/MedRAG, src/medgapqa/data/statpearls.py)
unchanged: section and subheading hierarchy is attached as the title, and paragraphs and lists under 200 characters are merged into the previous snippet up to 1,000 characters.
Changes: input and output paths are arguments, output is written to one file, and the license URL of each document is recorded.

Usage: python -I src/medgapqa/statpearls_chunk.py <nxml_dir> <out_jsonl>
The corpus and index are for internal use and not redistributed (most documents are CC BY-NC-ND 4.0).
"""
import json
import os
import sys
import xml.etree.ElementTree as ET


def ends_with_ending_punctuation(s):
    return s.endswith((".", "?", "!"))


def concat(title, content):
    if ends_with_ending_punctuation(title.strip()):
        return title.strip() + " " + content.strip()
    return title.strip() + ". " + content.strip()


def extract_text(element):
    text = (element.text or "").strip()
    for child in element:
        text += (" " if len(text) else "") + extract_text(child)
        if child.tail and len(child.tail.strip()) > 0:
            text += (" " if len(text) else "") + child.tail.strip()
    return text.strip()


def is_subtitle(element):
    if element.tag != "p" or len(list(element)) != 1 or list(element)[0].tag != "bold":
        return False
    tail = list(element)[0].tail
    return not (tail and len(tail.strip()) > 0)


def snippet(fname, j, title, content):
    return {"id": f"{fname}_{j}", "title": title, "content": content, "contents": concat(title, content)}


def extract(fpath):
    fname = os.path.basename(fpath).replace(".nxml", "")
    root = ET.parse(fpath).getroot()
    title = root.find(".//title").text
    lic = root.find(".//license")
    license_url = None
    if lic is not None:
        license_url = next((v for k, v in lic.attrib.items() if k.endswith("href")), None)
    saved = []
    j = 0
    for sec in root.findall(".//sec"):
        sec_title = sec.find("./title").text.strip()
        prefix = " -- ".join([title, sec_title])
        last_text = None
        last_json = None
        last_node = None
        for ch in sec:
            if is_subtitle(ch):
                last_text = None
                last_json = None
                sub_title = extract_text(ch)
                prefix = " -- ".join(prefix.split(" -- ")[:2] + [sub_title])
            elif ch.tag == "p":
                curr_text = extract_text(ch)
                if len(curr_text) < 200 and last_text is not None and len(last_text + curr_text) < 1000:
                    last_text = " ".join([last_json["content"], curr_text])
                    last_json = snippet(fname, 0, last_json["title"], last_text) | {"id": last_json["id"]}
                    saved[-1] = last_json
                else:
                    last_text = curr_text
                    last_json = snippet(fname, j, prefix, curr_text)
                    saved.append(last_json)
                    j += 1
            elif ch.tag == "list":
                list_text = [extract_text(c) for c in ch]
                if last_text is not None and len(" ".join(list_text) + last_text) < 1000:
                    last_text = " ".join([last_json["content"]] + list_text)
                    last_json = snippet(fname, 0, last_json["title"], last_text) | {"id": last_json["id"]}
                    saved[-1] = last_json
                elif len(" ".join(list_text)) < 1000:
                    last_text = " ".join(list_text)
                    last_json = snippet(fname, j, prefix, last_text)
                    saved.append(last_json)
                    j += 1
                else:
                    last_text = None
                    last_json = None
                    for c in list_text:
                        saved.append(snippet(fname, j, prefix, c))
                        j += 1
                if last_node is not None and is_subtitle(last_node):
                    prefix = " -- ".join([title, sec_title])
            last_node = ch
    for s in saved:
        s["license"] = license_url
    return saved


def main():
    src, out = sys.argv[1], sys.argv[2]
    fnames = sorted(f for f in os.listdir(src) if f.endswith(".nxml"))
    n_docs = n_snip = 0
    failed = []
    with open(out, "w") as f:
        for fn in fnames:
            try:
                snips = extract(os.path.join(src, fn))
            except Exception as e:  # documents that fail to parse are recorded and skipped
                failed.append((fn, repr(e)[:200]))
                continue
            if snips:
                n_docs += 1
            for s in snips:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
                n_snip += 1
    print(json.dumps({"files": len(fnames), "docs_with_snippets": n_docs, "snippets": n_snip,
                      "parse_failed": len(failed), "failed_examples": failed[:5]}))


if __name__ == "__main__":
    main()
