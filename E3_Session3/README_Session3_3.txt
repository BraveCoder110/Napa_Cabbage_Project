E3 Session 3.3 - Real-data CPU training smoke test

Copy E3_Session3 into Napa_Cabbage_Project (merge it with the existing folder).
Run from the Napa_Cabbage_Project root in PyCharm Terminal:

python.exe E3_Session3\train_smoke_test.py --images E3_Session2\Bok_Choy_Dataset.zip --masks E3_output\binary_masks.zip --output experiments\session3_3 --steps 4

Expected artifacts:
  experiments\session3_3\training_log.csv
  experiments\session3_3\unet_smoke_test.pth

The script uses CPU and directly streams image/mask pairs from ZIP files.
It trains on 4 mini-batches, not the whole dataset. Loss need not decrease
monotonically on different mini-batches. This is NOT the formal experiment.
