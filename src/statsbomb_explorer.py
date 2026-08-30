from statsbombpy import sb


def load_competitions():
    """
    Load all competitions and seasons available
    from StatsBomb Open Data.
    """
    competitions = sb.competitions()
    return competitions


if __name__ == "__main__":
    print("=" * 60)
    print("LIVEMATCH INTELLIGENCE")
    print("StatsBomb Data Explorer")
    print("=" * 60)

    print("\nLoading available football competitions...")

    competitions = load_competitions()

    print("Competition data loaded successfully!")
    print(f"Competition-season records: {len(competitions)}")

    print("\nLoading Bundesliga 2023/2024 matches...")

    matches = sb.matches(
        competition_id=9,
        season_id=281
    )

    print("Matches loaded successfully!")
    print(f"\nNumber of matches: {len(matches)}")

    print("\nAvailable match fields:")
    print(matches.columns.tolist())

    print("\nFirst 20 Bundesliga matches:\n")

    match_columns = [
        "match_id",
        "match_date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    ]

    print(
        matches[match_columns]
        .head(20)
        .to_string(index=False)
    )
    print("\nLoading match events...")

match_id = 3895309

events = sb.events(match_id=match_id)

print("Events loaded successfully!")

print(f"\nTotal events in match: {len(events)}")

print("\nAvailable event fields:")
print(events.columns.tolist())

print("\nEvent types in this match:\n")

print(
    events["type"]
    .value_counts()
    .to_string()
)