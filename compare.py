import json
import os
import re

json_path = r'c:\Users\Murtaza\Source\StarAI\src\Objects\Ships\space_ships.json'
uqm_ships_dir = r'c:\Users\Murtaza\Source\StarAI\uqm_src\sc2\src\uqm\ships'

ship_map = {
    "Androsynth": "androsyn",
    "Arilou": "arilou",
    "Chenjesu": "chenjesu",
    "Chmmr": "chmmr",
    "Druuge": "druuge",
    "Earthling": "human",
    "Ilwrath": "ilwrath",
    "KohrAh": "blackurq",
    "KzerZa": "urquan",
    "Mmrnmrhm": "mmrnmhrm",
    "Melnorme": "melnorme",
    "Mycon": "mycon",
    "Orz": "orz",
    "Pkunk": "pkunk",
    "Shofixti": "shofixti",
    "Slylandro": "probe",
    "Spathi": "spathi",
    "Supox": "supox",
    "Syreen": "syreen",
    "Thraddash": "thradd",
    "Umgah": "umgah",
    "Utwig": "utwig",
    "Vux": "vux",
    "Yehat": "yehat",
    "ZoqFotPik": "zoqfot"
}

with open(json_path, 'r') as f:
    ships_data = json.load(f)

for ship_name, folder_name in ship_map.items():
    c_file = os.path.join(uqm_ships_dir, folder_name, f"{folder_name}.c")
    if not os.path.exists(c_file):
        print(f"Skipping {ship_name}, {c_file} not found")
        continue
    
    with open(c_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        
    def get_define(name):
        # Match #define NAME value, handling comments or (value)
        m = re.search(r'#define\s+'+name+r'\s+(?:/\*.*?\*/\s*)?\(?([0-9]+)\)?', content)
        if m:
            return int(m.group(1))
        
        # Match negative numbers
        m2 = re.search(r'#define\s+'+name+r'\s+(?:/\*.*?\*/\s*)?\(?(-?[0-9]+)\)?', content)
        if m2:
            return int(m2.group(1))
        return None

    def get_super_melee_cost():
        m = re.search(r'(\d+),\s*/\*\s*Super Melee cost', content)
        if m:
            return int(m.group(1))
        return None

    starai = ships_data.get(ship_name)
    if not starai:
        continue
        
    discrepancies = []
    
    # Cost
    u_cost = get_super_melee_cost()
    s_cost = starai.get("cost")
    if u_cost is not None and s_cost != u_cost:
        discrepancies.append(f"  Cost: StarAI={s_cost}, UQM={u_cost}")
        
    # max_hp
    u_max_hp = get_define("MAX_CREW")
    s_max_hp = starai.get("max_hp")
    if u_max_hp is not None and s_max_hp != u_max_hp:
        discrepancies.append(f"  Max HP (Crew): StarAI={s_max_hp}, UQM={u_max_hp}")
        
    # max_energy
    u_max_en = get_define("MAX_ENERGY")
    s_max_en = starai.get("max_energy")
    if u_max_en is not None and s_max_en != u_max_en:
        discrepancies.append(f"  Max Energy: StarAI={s_max_en}, UQM={u_max_en}")

    # thrust
    u_thrust = get_define("MAX_THRUST")
    s_thrust = starai.get("max_thrust")
    if u_thrust is not None and s_thrust != u_thrust:
        discrepancies.append(f"  Max Thrust: StarAI={s_thrust}, UQM={u_thrust}")

    # thrust increment
    u_ti = get_define("THRUST_INCREMENT")
    s_ti = starai.get("thrust_increment")
    if u_ti is not None and s_ti != u_ti:
        discrepancies.append(f"  Thrust Increment: StarAI={s_ti}, UQM={u_ti}")

    # turn wait
    u_tw = get_define("TURN_WAIT")
    s_tw = starai.get("turn_wait")
    if u_tw is not None and s_tw != u_tw:
        discrepancies.append(f"  Turn Wait: StarAI={s_tw}, UQM={u_tw}")

    # mass
    u_mass = get_define("SHIP_MASS")
    s_mass = starai.get("mass")
    if u_mass is not None and s_mass != u_mass:
        discrepancies.append(f"  Mass: StarAI={s_mass}, UQM={u_mass}")

    # weapon cost
    u_a1 = get_define("WEAPON_ENERGY_COST")
    s_a1 = starai.get("a1_cost")
    if u_a1 is not None and s_a1 != u_a1:
        discrepancies.append(f"  A1 Cost: StarAI={s_a1}, UQM={u_a1}")

    # weapon wait
    u_a1w = get_define("WEAPON_WAIT")
    s_a1w = starai.get("a1_wait")
    if u_a1w is not None and s_a1w != u_a1w:
        discrepancies.append(f"  A1 Wait: StarAI={s_a1w}, UQM={u_a1w}")

    # special cost
    u_a2 = get_define("SPECIAL_ENERGY_COST")
    s_a2 = starai.get("a2_cost")
    if u_a2 is not None and s_a2 != u_a2:
        discrepancies.append(f"  A2 Cost: StarAI={s_a2}, UQM={u_a2}")

    # special wait
    u_a2w = get_define("SPECIAL_WAIT")
    s_a2w = starai.get("a2_wait")
    if u_a2w is not None and s_a2w != u_a2w:
        discrepancies.append(f"  A2 Wait: StarAI={s_a2w}, UQM={u_a2w}")

    if discrepancies:
        print(f"\n--- {ship_name} ---")
        for d in discrepancies:
            print(d)

