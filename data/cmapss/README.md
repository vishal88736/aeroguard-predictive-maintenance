# NASA C-MAPSS input files

Place the official files here, or import the official ZIP through the website:

```text
train_FD001.txt  test_FD001.txt  RUL_FD001.txt
train_FD002.txt  test_FD002.txt  RUL_FD002.txt
train_FD003.txt  test_FD003.txt  RUL_FD003.txt
train_FD004.txt  test_FD004.txt  RUL_FD004.txt
```

Download manually from https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data .
No dataset is downloaded when the notebook is opened or the website starts.
Training only reads `train_*.txt`. Official test data and endpoint truth are read by the separate final evaluator.
