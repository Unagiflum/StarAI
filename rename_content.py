import os

files_to_update = [
    r"c:\Users\Murtaza\Source\StarAI\src\Objects\Ships\ZoqFotPik\ZoqFotPik.py",
    r"c:\Users\Murtaza\Source\StarAI\src\Objects\Ships\ZoqFotPik\A1\ZoqFotPikA1.py",
    r"c:\Users\Murtaza\Source\StarAI\src\Objects\Ships\ZoqFotPik\A2\ZoqFotPikA2.py",
    r"c:\Users\Murtaza\Source\StarAI\tests\test_zoqfotpik_a2.py",
    r"c:\Users\Murtaza\Source\StarAI\tests\test_ship_actions_characterization.py",
    r"c:\Users\Murtaza\Source\StarAI\src\Objects\Ships\space_ships.json",
    r"c:\Users\Murtaza\Source\StarAI\src\Objects\Ships\abilities.json",
    r"c:\Users\Murtaza\Source\StarAI\src\Config\fleets.json"
]

for filepath in files_to_update:
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        new_content = content.replace("ZoqFot", "ZoqFotPik")
        
        if new_content != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Updated content in: {filepath}")
    else:
        print(f"File not found: {filepath}")

