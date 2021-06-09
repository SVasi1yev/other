from tqdm import tqdm
import json


def add_dialogs(file, list_):
    with open(file) as f:
        persona1 = []
        persona2 = []
        replicas = []
        lines = f.readlines()
        i = 0
        flag = False
        for i in tqdm(range(len(lines))):
            line = ' '.join(lines[i].split(' ')[1:])
            if line.startswith('your persona:'):
                if flag:
                    list_.append([persona1, persona2, replicas])
                    flag = False
                    persona1 = []
                    persona2 = []
                    replicas = []
                persona1.append(line.split(': ')[1][:-1])
            elif line.startswith('partner\'s persona: '):
                persona2.append(line.split(': ')[1][:-1])
            else:
                flag = True
                splited = line.split('\t')
                replicas.append(splited[0])
                replicas.append(splited[1])


personachat = []
add_dialogs('personachat/train_both_original.txt', personachat)
add_dialogs('personachat/valid_both_original.txt', personachat)
add_dialogs('personachat/test_both_original.txt', personachat)

with open('personachat.json', 'w') as f:
    json.dump(personachat, f)
