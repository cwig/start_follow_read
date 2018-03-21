import json
import sys

from datadiff import diff

with open(sys.argv[1]) as f:
    data1 = json.load(f)

with open(sys.argv[2]) as f:
    data2 = json.load(f)


for i in range(len(data1)):
    for l1, l2 in zip(data1[i]['lf'], data2[i]['lf']):
        assert l1['x0'] == l2['x0']
        assert l1['x1'] == l2['x1']
        assert l1['y0'] == l2['y0']
        assert l1['y1'] == l2['y1']
