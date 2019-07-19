import pyxro
import requests

sample = pyxro.MultilayerSample()
sample.from_parfile('Template.par')
converted_data = sample.to_parfile()

# with open('output.par', 'w') as f:
    # f.write(converted_data)

teste = pyxro.MultilayerSample()
teste.from_par(converted_data)

with open('Template.par', 'r') as f:
    data = f.read()

res = requests.post('http://localhost:5000/yxro/par2json', json=data)
print(res.content.decode('utf-8'))
