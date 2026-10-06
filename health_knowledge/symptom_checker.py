import json
from pathlib import Path

BASE_DIR = Path(__file__).parent


def load_condition(filename):
    path = BASE_DIR / filename

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_symptoms(user_symptoms, condition):

    primary_symptoms = {
        symptom.lower()
        for symptom in condition.get("symptoms", [])
    }

    supporting_symptoms = {
        symptom.lower()
        for symptom in condition.get("supporting_symptoms", [])
    }

    all_symptoms = primary_symptoms | supporting_symptoms

    user_symptoms = {
        symptom.lower()
        for symptom in user_symptoms
    }

    matched_primary = user_symptoms.intersection(
        primary_symptoms
    )

    matched_supporting = user_symptoms.intersection(
        supporting_symptoms
    )

    matched = matched_primary | matched_supporting

    if all_symptoms:
        score = len(matched) / len(all_symptoms)
    else:
        score = 0

    return {
        "condition": condition["condition"],
        "matched_symptoms": sorted(matched),
        "primary_matches": sorted(matched_primary),
        "supporting_matches": sorted(matched_supporting),
        "match_score": round(score * 100, 1),
        "self_care": condition["self_care"],
        "red_flags": condition["red_flags"],
        "seek_medical_care": condition["seek_medical_care"],
        "disclaimer": condition["disclaimer"]
    }


if __name__ == "__main__":

    condition = load_condition("common_cold.json")

    test_symptoms = [
        "runny nose",
        "sore throat",
        "cough",
        "mild headache"
    ]

    result = check_symptoms(
        test_symptoms,
        condition
    )

    print("Condition:", result["condition"])
    print("Match:", result["match_score"], "%")
    print("Matched symptoms:", result["matched_symptoms"])

    print("\nPrimary matches:")
    for item in result["primary_matches"]:
        print("-", item)

    print("\nSupporting matches:")
    for item in result["supporting_matches"]:
        print("-", item)

    print("\nSelf-care:")
    for item in result["self_care"]:
        print("-", item)

    print("\nRed flags:")
    for item in result["red_flags"]:
        print("-", item)