# NSGPy

To install,
```console
foo@bar:~/NSGPy$ python setup.py install
```

Basic usage,
```python
from NSGPy.NumPy import LLS
model = LLS(input_dim=2)
model.fit(X, y)
mean, cov = model.predict(X_new, return_cov=True, diag=False)
```
Editor notebook with few experiments is located [here](https://github.com/patel-zeel/NSGPy/blob/main/dev/LLSModelEditor.ipynb)
