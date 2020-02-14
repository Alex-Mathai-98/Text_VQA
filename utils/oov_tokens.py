import pickle
import ast

f = open('tokens.txt', 'r')

all_new_tokens = []

def read_tokens(_list):
    new_tokens = []

    flag = 0
    word = ''

    for token in _list:
        if flag:
            if not token.startswith('##'):
                print (word)
                word = token + word
                print (word)
                new_tokens.append(word)
                flag = 0
                word = ''
            else:
                word = token[2:] + word
        else:
            if token.startswith('##'):
                word = token[2:]
                flag = 1

    return new_tokens

for line in f.readlines():
    _list = ast.literal_eval(line)
    _list.reverse()
    all_new_tokens += read_tokens(_list)

f.close()

with open('oov_tokens.pkl', 'wb') as f:
    pickle.dump(all_new_tokens, f)

# lis = ['what', 'does', 'this', 'jet', 'ad', '##vert', '##ise', 'as', 'having', '?']

# lis.reverse()

# print (read_tokens(lis))