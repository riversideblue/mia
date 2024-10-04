import json

with open(f"results.json","w") as f:
    json.dump({},f,indent=1)
    results = json.load(open(f"results.json"))
    json.dump(results,f)
