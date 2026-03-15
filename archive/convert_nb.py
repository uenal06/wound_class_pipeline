import json
with open('c:/Users/Mustafa/Desktop/wound_pipeline/archive/Classification_Mask_New_1.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('c:/Users/Mustafa/Desktop/wound_pipeline/archive/Classification_Mask_New_1.py', 'w', encoding='utf-8') as f:
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            f.write("".join(cell['source']))
            f.write("\n\n")
