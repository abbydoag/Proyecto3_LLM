import xml.etree.ElementTree as ET
from lxml import etree
import mwparserfromhell
from bs4 import BeautifulSoup
import json
import re
import os

RAW_PATH = "data/raw/StardewDump.xml"
OUT_PATH = "data/processed/Stardew_data.json"

GIFT_SECTIONS = {"Love", "Like", "Neutral", "Dislike", "Hate"}

NOISE_PATTERNS = [
    r"\bMining\b", r"\bCooking\b", r"\bFarming\b", r"\bForaging\b",
    r"\bFishing\b", r"\bFurnace\b", r"\bCheese Press\b", r"\bKeg\b",
    r"\bPig\b", r"\bPigs\b", r"\bDuck\b", r"\bDucks\b", r"\bRabbit\b",
    r"\bRabbits\b", r"\bSheep\b", r"\bCow\b", r"\bGoat\b", r"\bChicken\b",
    r"\bVoid Chicken\b", r"\bTapper\b", r"\bLoom\b", r"\bPreserves Jar\b",
    r"\bMayonnaise Machine\b", r"\bOil Maker\b", r"\bBee House\b",
    r"\bBookseller\b", r"\bMonsters\b", r"\bArtifacts\b", r"\bTrinkets\b",
    r"\bBooks\b", r"\bMagma Geode\b", r"\bOmni Geode\b", r"\bFrozen Geode\b",
    r"\bThe Stardrop Saloon\b", r"\bPierre's General Store\b",
    r"\bThe Beach Resort\b", r"\bGinger Island\b", r"\bVolcano Dungeon\b",
    r"\bThe Mines\b", r"\bThe Desert\b", r"\bSecret Woods\b",
    r"\bForaging - \w+\b", r"\bMinerals\b", r"\bCrops\b",
    r"\bFarming - \w+\b", r"\bTilling\b",
    r"\bFall\b", r"\bSpring\b", r"\bSummer\b", r"\bWinter\b",
    r"\bGeode\b", r"\bTrash\b", r"\bCrab Pot\b", r"\bCrab Pots\b",
    r"\bFish Pond\b", r"\bLightning Rod\b", r"\bSolar Panel\b",
    r"\bPomegranate Tree\b", r"\bThe Cave\b"
]

TECHNICAL_PAGES = {
    "Mining", "Foraging", "Cooking", "Fish", "Crops", "Friendship", "The Mines", "Skull Cavern",
    "Volcano Dungeon", "Quarry", "Quarry Mine", "Artisan Goods", "Fertilizer", "Bait", "Tackle", "Weapons", "Hats",
    "Footwear", "Rings", "Books", "Artifacts", "Minerals", "Bundles", "Secret Notes", "Crafting", "Furniture",
    "Wallpaper", "Flooring", "Fishing", "Trees", "Farm Maps", "Skills", "Achievements"
}


def _extract_infobox_favorites(raw_text: str, title: str) -> list[dict]:
    #chunk infobox (formato)
    chunks = []
    m = re.search(
        r"\|\s*favorites\s*=\s*(.+?)(?=\n\s*\|[a-z]|\n\}\})",
        raw_text, re.DOTALL
    )
    if not m:
        return chunks
    raw = m.group(1)
    items = re.findall(r"\{\{\s*Name\s*\|\s*([^}|]+)", raw, re.IGNORECASE)
    items = [i.strip() for i in items if i.strip()]
    if not items:
        return chunks
    seen = set()
    uniq = []
    for it in items:
        k = it.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(it)
    text = f"{title} loved gifts (favorite items). {title} loves: " + ", ".join(uniq) + "."
    chunks.append({
        "page": title,
        "section": "Love",
        "text": text,
    })
    return chunks


def clean_wikitext(raw_text: str) -> str:
    wikicode = mwparserfromhell.parse(raw_text)
    for template in wikicode.filter_templates():
        if template.name.strip().lower() == "name":
            try:
                wikicode.replace(template, str(template.get(1).value).strip())
            except Exception:
                pass

    text = str(wikicode)

    def table_to_text(match):
        table = match.group(0)
        if "!Time" in table and "!Location" in table:
            return ""
        lines = table.split("\n")
        is_data_table = "!Name" in table or "!Image" in table or "!Description" in table
        headers = []
        if is_data_table:
            for line in lines:
                if line.strip().startswith("!"):
                    h = re.sub(r"!|<[^>]+>|style=\"[^\"]*\"|colspan=\"\d+\"|rowspan=\"\d+\"", "", line)
                    h = h.strip().strip("|").strip()
                    if h:
                        headers.append(h)
        cells = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("{|") or line.startswith("|}") or line == "|-":
                continue
            if not (line.startswith("|") or line.startswith("!")):
                continue
            content = line[1:].strip()
            if len(re.findall(r"\{\{\s*Name\s*\|", content, flags=re.IGNORECASE)) >= 2:
                continue
            if "|" in content:
                parts = re.split(r"\|(?![^\[]*\]\])(?![^{]*\}\})", content, maxsplit=1)
                content = parts[1].strip() if len(parts) > 1 else parts[0].strip()
            content = re.sub(r"\[\[File:[^\]]*\]\]", "", content)
            content = re.sub(r"<[^>]+>", " ", content)
            content = re.sub(r"\[\[.*?\|(.+?)\]\]", r"\1", content)
            content = re.sub(r"\[\[(.+?)\]\]", r"\1", content)
            content = re.sub(r"''+", "", content)
            content = re.sub(r"\{\{Description\|[^}]*\}\}", "", content)
            content = re.sub(r"^\s*\|+\s*", "", content)
            content = re.sub(r"\s*\|+\s*$", "", content)
            content = re.sub(r"\s+", " ", content).strip()
            if content and len(content) > 1 and not line.strip().startswith("!"):
                cells.append(content)
                if is_data_table and headers and len(cells) >= len(headers):
                    rows = []
                    for i in range(0, len(cells), len(headers)):
                        row_cells = cells[i:i + len(headers)]
                        if len(row_cells) > 0 and row_cells[0]:
                            is_recipe = any("Ingredient" in h or "Recipe" in h for h in headers)
                            prefix = "Recipe" if is_recipe else "Item"
                            row_str = f"{prefix}: " + ". ".join(
                                [f"{headers[j]} is {row_cells[j]}"
                                 for j in range(min(len(headers), len(row_cells)))
                                 if row_cells[j]]
                            ) + "."
                            rows.append(row_str)
                    return "\n".join(rows) + "\n"
        return " | ".join(cells)

    #tablas
    text = re.sub(r"\{\|.*?\|\}", table_to_text, text, flags=re.DOTALL)

    #templates limpieza
    text = re.sub(r"\{\{[^{}]*\}\}", "", text, flags=re.DOTALL)
    text = re.sub(r"\[\[File:[^\]]*\]\]", "", text)
    text = re.sub(r"\[\[.*?\|(.+?)\]\]", r"\1", text)
    text = re.sub(r"\[\[(.+?)\]\]", r"\1", text)
    text = re.sub(r"'''?", "", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^>]*/>", "", text)
    text = re.sub(r"<[^>]+>", " ", text)
    #limpieza cont
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"^\s*\|[-}].*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\|.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def is_useful_chunk(text: str, min_ratio: float = 0.4, page: str = "") -> bool:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return False

    if page in TECHNICAL_PAGES:
        pipe_lines = sum(1 for l in lines if l.count("|") >= 3)
        if pipe_lines / len(lines) > 0.85:
            return False
        return True

    time_lines = sum(1 for l in lines if re.search(r"\d{1,2}:\d{2}\s*(AM|PM)", l))
    if time_lines / len(lines) > 0.5:
        return False
    pipe_lines = sum(1 for l in lines if "|" in l)
    if pipe_lines / len(lines) > 0.6:
        return False
    useful = sum(1 for l in lines if len(l.split()) > 6 or l.endswith("."))
    return (useful / len(lines)) >= min_ratio


def build_prefix(title: str, section: str) -> str:
    if section.startswith("Universal "):
        return f"Stardew Valley {section}: "
    if section == "Love":
        return f"{title} loved gifts (favorite items). {title} loves: "
    elif section == "Like":
        return f"{title} liked gifts. {title} likes: "
    elif section == "Neutral":
        return f"{title} neutral gifts. {title} is neutral about: "
    elif section == "Dislike":
        return f"{title} disliked gifts. {title} dislikes: "
    elif section == "Hate":
        return f"{title} hated gifts. {title} hates: "
    elif section == "General":
        return f"{title} - General: "
    else:
        return f"{title} - {section}: "


def _clean_item_list(raw: str) -> list[str]:
    raw = raw.replace("|", "\n")
    lines = [l.strip() for l in raw.split("\n") if l.strip()]

    items = []
    for line in lines:
        if any(re.search(p, line) for p in NOISE_PATTERNS):
            continue
        line_clean = re.sub(r"\s*\([^)]*\)\s*$", "", line).strip()
        if not line_clean:
            continue
        if re.match(r"^Any \w+", line_clean):
            continue
        if line_clean.startswith("("):
            continue

        has_space = " " in line_clean
        no_spaces = line_clean.replace(" ", "")
        transitions = len(re.findall(r"[a-z][A-Z]", no_spaces))
        if not has_space and transitions >= 1:
            continue
        if has_space and transitions >= 2:
            continue

        if re.search(r"(Flour|Sugar|Egg|Milk|Oil|Rice|Coal|Ore|Vinegar)$", line_clean):
            continue
        if len(line_clean) > 60:
            continue
        if line_clean.endswith(".") and len(line_clean.split()) > 3:
            continue
        if re.search(r"\b(Monster drop|Drop|Drops)\b", line_clean):
            continue

        items.append(line_clean)

    seen = set()
    unique = []
    for it in items:
        k = it.lower()
        if k not in seen:
            seen.add(k)
            unique.append(it)
    return unique


def extract_gift_items(text: str, title: str, section: str) -> str:
    prefix = build_prefix(title, section)
    if text.startswith(prefix):
        text = text[len(prefix):]

    universal_label = ""
    label_match = re.search(r"(All Universal \w+(?:\s*\([^)]*\))?)", text)
    if label_match:
        universal_label = label_match.group(1).strip()

    text_wo_label = text
    if label_match:
        text_wo_label = text[:label_match.start()] + text[label_match.end():]

    items = _clean_item_list(text_wo_label)

    if items:
        result = prefix + ", ".join(items) + ". "
    else:
        result = prefix

    if universal_label:
        if items:
            result += f"Also: {universal_label}."
        else:
            result += f"({universal_label})."

    return result.strip()


def extract_chunks(text: str, title: str, max_chars: int = 800) -> list[dict]:
    chunks = []
    section_pattern = re.compile(r"(={2,})\s*(.+?)\s*\1")
    parts = section_pattern.split(text)

    def add_content(content: str, section_name: str):
        content = content.strip()
        if not content or len(content) < 20:
            return

        if section_name in GIFT_SECTIONS:
            cleaned = extract_gift_items(
                build_prefix(title, section_name) + content,
                title, section_name
            )
            if len(cleaned) >= 20:
                chunks.append({
                    "page": title,
                    "section": section_name,
                    "text": cleaned
                })
            return

        limit = 800 if title in TECHNICAL_PAGES else max_chars

        if len(content) <= limit and title not in TECHNICAL_PAGES:
            chunks.append({
                "page": title,
                "section": section_name,
                "text": build_prefix(title, section_name) + content
            })
        else:
            overlap = 100
            start = 0
            while start < len(content):
                end = start + limit
                if end < len(content):
                    next_space = content.find(" ", end)
                    next_newline = content.find("\n", end)
                    candidates = [c for c in (next_space, next_newline) if c != -1]
                    if candidates:
                        end = min(candidates)
                fragment = content[start:end].strip()

                if len(fragment) >= 20:
                    enriched_text = f"{title}-- {section_name} - {fragment}"
                    chunks.append({
                        "page": title,
                        "section": section_name,
                        "text": enriched_text
                    })
                start = end - overlap

    if parts:
        add_content(parts[0], "General")

    i = 1
    while i + 2 < len(parts):
        raw_title = parts[i + 1]
        section_name = re.sub(r"\s+", " ", raw_title).strip()
        content = parts[i + 2]
        add_content(content, section_name)
        i += 3

    return chunks


def main():
    if not os.path.exists(RAW_PATH):
        print(f"No se encontro {RAW_PATH}")
        return

    print(f"Parseando {RAW_PATH}")

    parser = etree.XMLParser(recover=True, huge_tree=True)
    tree = etree.parse(RAW_PATH, parser)
    root = tree.getroot()

    ns_match = re.match(r"\{(.+?)\}", root.tag)
    ns = f"{{{ns_match.group(1)}}}" if ns_match else ""
    print(f"Namespace detectado: {ns}")

    all_chunks = []

    for page in root.findall(f".//{ns}page"):
        title_el = page.find(f"{ns}title")
        if title_el is None or title_el.text is None:
            continue
        title = title_el.text.strip().replace("_"," ")

        revision = page.find(f"{ns}revision")
        if revision is None:
            continue
        text_el = revision.find(f"{ns}text")
        if text_el is None or not text_el.text:
            continue

        raw = text_el.text

        infobox_chunks = _extract_infobox_favorites(raw, title)
        if infobox_chunks:
            all_chunks.extend(infobox_chunks)

        clean_text = clean_wikitext(raw)

        if len(clean_text) < 50:
            print(f"{title}: texto corto omitido")
            continue

        chunks = extract_chunks(clean_text, title)
        all_chunks.extend(chunks)
        if infobox_chunks:
            print(f"- {title}: {len(chunks)} chunks (+{len(infobox_chunks)} Infobox), {len(clean_text)} caracteres")
        else:
            print(f"- {title}: {len(chunks)} chunks, {len(clean_text)} caracteres")

    if not all_chunks:
        print("No se generaron chunks, revisar XML o namespace")
        return

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    avg = sum(len(c["text"]) for c in all_chunks) / len(all_chunks)
    print(f"\nXML a JSON completado ({len(all_chunks)} chunks en {OUT_PATH})")
    print(f"Tamano prom: {avg:.0f} caracteres")


if __name__ == "__main__":
    main()