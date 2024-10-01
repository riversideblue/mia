import os
import json

# os.environの内容を整形してJSONファイルに保存
with open('environment_variables.json', 'w') as f:
    json.dump(dict(os.environ), f, indent=4)

