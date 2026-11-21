import re
from pathlib import Path

project_root = Path(__file__).resolve().parent
files_to_update = [
    "src/Objects/Ships/ZoqFotPik/ZoqFotPik.py",
    "src/Objects/Ships/ZoqFotPik/A1/ZoqFotPikA1.py",
    "src/Objects/Ships/ZoqFotPik/A2/ZoqFotPikA2.py",
    "tests/test_zoqfotpik_a2.py",
    "tests/test_ship_actions_characterization.py",
    "src/Objects/Ships/space_ships.json",
    "src/Objects/Ships/abilities.json",
    "src/Config/fleets.json",
]

for relative_path in files_to_update:
    filepath = project_root / relative_path
    if filepath.exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        new_content = re.sub(r"ZoqFot(?!Pik)", "ZoqFotPik", content)
        
        if new_content != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Updated content in: {filepath}")
    else:
        print(f"File not found: {filepath}")

