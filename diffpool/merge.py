color1 = 'green'
color2 = 'yellow'
month = '1904'

f_1 = open('taxi/' + color1 + '_' + month + '.txt', 'r')

f_2 = open('taxi/' + color2 + '_' + month + '.txt', 'r')

g = open('taxi/' + color1 + '_' + color2 + '_' + month + '.txt', 'w')

data= dict()

for line in f_1:
    lines = line.split(',')
    key = lines[2]

    if not key in data.keys():
        data[key] = []
    data[key].append(line)
print('done with ' + color1)

for line in f_2:
    lines = line.split(',')
    key = lines[2]

    if not key in data.keys():
        data[key] = []
    data[key].append(line)
print('done with ' + color2)

datehours = data.keys()
keys = sorted(list(datehours))

for key in keys:
    for edge in data[key]:
        g.write(edge)


f_1.close()
f_2.close()
g.close()