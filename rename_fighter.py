import os
import re

directories = ['src', 'tests']

def replace_cases(text):
    text = re.sub(r'FighterCollisionCapabilities', 'SpecialObjectCollisionCapabilities', text)
    text = re.sub(r'fighter_collision_capabilities', 'special_object_collision_capabilities', text)
    
    # Plural
    text = re.sub(r'\bFighters\b', 'SpecialObjects', text)
    text = re.sub(r'\bfighters\b', 'special_objects', text)
    text = re.sub(r'\bFIGHTERS\b', 'SPECIAL_OBJECTS', text)
    
    # Singular
    text = re.sub(r'\bFighter\b', 'SpecialObject', text)
    text = re.sub(r'\bfighter\b', 'special_object', text)
    text = re.sub(r'\bFIGHTER\b', 'SPECIAL_OBJECT', text)
    
    return text

for d in directories:
    for root, _, files in os.walk(d):
        for file in files:
            if file.endswith('.py') or file.endswith('.json'):
                path = os.path.join(root, file)
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                new_content = replace_cases(content)
                
                if new_content != content:
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f'Updated {path}')
