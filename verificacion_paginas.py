import re
from lxml import etree

RAW_PATH = "data/raw/StardewDump.xml"

parser = etree.XMLParser(recover=True, huge_tree=True)
tree = etree.parse(RAW_PATH, parser)
root = tree.getroot()

ns_match = re.match(r"\{(.+?)\}", root.tag)
ns = f"{{{ns_match.group(1)}}}" if ns_match else ""

titles = []
for p in root.findall(f".//{ns}page"):
    title_el = p.find(f"{ns}title")
    if title_el is not None and title_el.text:
        titles.append(title_el.text.strip())

print(f"Total paginas en XML: {len(titles)}")
print()

pages_check = [
    #lista modificable
    "Mining", "Copper_Ore", "Gold_Ore", "Iridium_Ore",
    "Crops", "Strawberry", "Cauliflower",
    "Pufferfish", "Fish", "Legend",
    "Friendship", "Abigail", "Sebastian", "Rusty_Key",
    "Marnie's_Ranch", "Wizard's_Tower",
    "Quarry_Mine", "Weapons", "Galaxy_Sword"
]

print("estan el XML?")
for p in pages_check:
    status = "Si" if p in titles else "No"
    print(f"  {p:15s} - |{status}|")

print("\nPasar a espacios:")
for p in pages_check:
    variants = [p, p.replace("_", " ")]
    for v in variants:
        if v in titles and v != p:
            print(f"  '{p}' => '{v}'")