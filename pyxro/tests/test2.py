import pyxro
import requests
import pprint

sample = pyxro.MultilayerSample()
sample.from_parfile('Template.par')

res = requests.post('http://localhost:5000/yxro/par2json', json=sample.to_parfile())

with open('lala', 'w') as f:
    f.write(res.content.decode('utf-8'))
