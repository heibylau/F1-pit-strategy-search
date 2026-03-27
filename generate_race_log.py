import json

def generate_race_log(path, filename):
    race_log = []
    for state in path:
        race_log.append({
            "lap": state[0],
            "compound": state[1],
            "tire_age": state[2],
            "total_time": state[3].item()
        })
    with open(filename, "w") as f:
        json.dump(race_log, f, indent=4)