# Progress Log

One line per session: what was done.

- [2026-06-19] [30 min] Bootstrapped src/ pipeline skeleton; src/preprocessing.py with load_skab_single / load_anomaly_free / load_skab_all / chronological_split. Split is per-source-file 80/20 (not a single global time cut) since each SKAB CSV is an independent experiment with its own clock — caught and fixed a bug where filenames collided across valve1/other/valve2 folders (e.g. both valve1/0.csv and valve2/0.csv -> "0"), which silently merged unrelated experiments before the fix. Verified via notebooks/01_explore.ipynb (executes end-to-end): anomaly-free=9401 rows/8 cols, 34 distinct fault-run CSVs=37459 rows, train/test anomaly rates 0.367/0.30. Cloned SensorGuard (hackathon repo) into ./SensorGuard as a git-ignored reference for porting anomaly taxonomy + feature logic next session. requirements.txt pinned to what's already installed in env/.
