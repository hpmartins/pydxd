import pyxro

sample = pyxro.MultilayerSample()
sample.from_parfile('test_mirror.par')

# sample.to_parfile() -> back to old parameter
# sample.to_json()    -> to json object