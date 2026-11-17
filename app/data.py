"""Create reproducible synthetic football commentary pairs."""
import json
import random
from pathlib import Path


TEAMS = ["Falcons", "Rovers", "United", "City", "Wolves", "Eagles"]

TEMPLATES = {
    "goal": [
        "Goal for {team}! The score is now {home} {hs}-{as_} {away}.",
        "{team} score! That makes it {home} {hs}-{as_} {away}.",
        "It's a goal for {team}! {home} {hs}-{as_} {away}.",
    ],
    "save": [
        "A save by {team}! The score stays {home} {hs}-{as_} {away}.",
        "{team} make the save! It remains {home} {hs}-{as_} {away}.",
        "Saved by {team}! Still {home} {hs}-{as_} {away}.",
    ],
    "full-time": [
        "Full-time! The final score is {home} {hs}-{as_} {away}.",
        "The match is over! {home} {hs}-{as_} {away}.",
        "That's full-time: {home} {hs}-{as_} {away}.",
    ],
}


def make_dataset(output_dir="data", seed=17, per_type=160):
    if not 10 <= per_type <= 1000:
        raise ValueError("per_type must be between 10 and 1000")

    rng = random.Random(seed)
    splits = {"train": [], "validation": [], "test": []}
    seen = set()

    for event_type in TEMPLATES:
        records = []

        while len(records) < per_type:
            home, away = rng.sample(TEAMS, 2)
            hs, away_score = rng.randrange(5), rng.randrange(5)
            minute = 90 if event_type == "full-time" else rng.randrange(1, 90)
            team = ""

            if event_type == "goal":
                team = rng.choice([home, away])
                # The supplied score is AFTER this goal.
                if team == home:
                    hs = max(1, hs)
                else:
                    away_score = max(1, away_score)
            elif event_type == "save":
                team = rng.choice([home, away])

            if event_type == "goal":
                event = (
                    f"Football. Minute {minute}. Goal for {team}. "
                    f"Score after goal: {home} {hs}-{away_score} {away}."
                )
            elif event_type == "save":
                event = (
                    f"Football. Minute {minute}. Save by {team}. "
                    f"Score unchanged: {home} {hs}-{away_score} {away}."
                )
            else:
                event = (
                    f"Football. Full-time. "
                    f"Final score: {home} {hs}-{away_score} {away}."
                )

            if event in seen:
                continue
            seen.add(event)

            commentary = rng.choice(TEMPLATES[event_type]).format(
                team=team,
                home=home,
                away=away,
                hs=hs,
                as_=away_score,
            )

            records.append({
                "event_type": event_type,
                "event": event,
                "commentary": commentary,
                "home": home,
                "away": away,
                "home_score": hs,
                "away_score": away_score,
                "team": team,
                "minute": minute,
            })

        rng.shuffle(records)
        train_end = int(per_type * 0.75)
        val_end = train_end + int(per_type * 0.125)

        splits["train"].extend(records[:train_end])
        splits["validation"].extend(records[train_end:val_end])
        splits["test"].extend(records[val_end:])

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    for name, rows in splits.items():
        rng.shuffle(rows)
        path = destination / f"{name}.jsonl"
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    return splits