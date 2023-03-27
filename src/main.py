from data.data import Data
import json 

with open('/home/kishore/workspace/Image-Captioning/parameters/params.json', 'r') as j:
    params = json.load(j)

dataset = Data(params)

dataset()