E3 Session 3.2: Lightweight U-Net CPU smoke test

Files:
  models/unet.py       Custom U-Net with English comments/docstrings
  test_unet.py         Forward/backward pass checks on random tensors

Place E3_Session3 in the Napa_Cabbage_Project root directory.
Run from PyCharm terminal with the working directory set to project root:

  python.exe E3_Session3\test_unet.py

This test DOES NOT train on images or calculate Dice/IoU.
Next session will connect the verified DataLoader and actual masks.
