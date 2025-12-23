"""Test NFL Data Library installation."""

import nfl_data_py as nfl

print("=" * 60)
print("NFL DATA LIBRARY TEST")
print("=" * 60)

print("\nImporting 2024 schedule...")
schedules = nfl.import_schedules([2024])

print(f"\nTotal games: {len(schedules)}")
print(f"\nColumns ({len(schedules.columns)}):")
print("-" * 40)

for col in schedules.columns:
    print(f"  - {col}")

print("\n" + "=" * 60)
print("SAMPLE DATA (First 5 rows)")
print("=" * 60)
print(schedules[['game_id', 'home_team', 'away_team', 'home_score', 'away_score']].head())

print("\n" + "=" * 60)
print("NFL DATA LIBRARY INSTALLED SUCCESSFULLY!")
print("=" * 60)
