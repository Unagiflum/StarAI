import os

def rename_in_dir(path):
    if not os.path.exists(path):
        print(f"Path does not exist: {path}")
        return
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if 'ZoqFot' in name:
                old_path = os.path.join(root, name)
                new_name = name.replace('ZoqFot', 'ZoqFotPik')
                new_path = os.path.join(root, new_name)
                os.rename(old_path, new_path)
                print(f"Renamed file: {old_path} -> {new_path}")
        for name in dirs:
            if 'ZoqFot' in name:
                old_path = os.path.join(root, name)
                new_name = name.replace('ZoqFot', 'ZoqFotPik')
                new_path = os.path.join(root, new_name)
                os.rename(old_path, new_path)
                print(f"Renamed directory: {old_path} -> {new_path}")

    # Finally rename the base directory if needed
    if 'ZoqFot' in os.path.basename(path):
        dir_name = os.path.dirname(path)
        base_name = os.path.basename(path)
        new_base = base_name.replace('ZoqFot', 'ZoqFotPik')
        new_path = os.path.join(dir_name, new_base)
        os.rename(path, new_path)
        print(f"Renamed base directory: {path} -> {new_path}")

base_dir = r"c:\Users\Murtaza\Source\StarAI\src\Objects\Ships\ZoqFot"
rename_in_dir(base_dir)

# Also rename the test file
test_file_old = r"c:\Users\Murtaza\Source\StarAI\tests\test_zoqfot_a2.py"
test_file_new = r"c:\Users\Murtaza\Source\StarAI\tests\test_zoqfotpik_a2.py"
if os.path.exists(test_file_old):
    os.rename(test_file_old, test_file_new)
    print(f"Renamed test file: {test_file_old} -> {test_file_new}")
