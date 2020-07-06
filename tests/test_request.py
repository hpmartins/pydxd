import requests

with open('test_generic.par', 'r') as f:
    parfile = f.read()

json_data = {'parfile': parfile}
res = requests.post('http://localhost:5000/yxro/par2json', json = json_data)
print(res.content.decode('utf-8'))
