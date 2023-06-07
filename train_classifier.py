import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb'))


# print(len(data_dict['data']))

# print(data_dict['labels'])

# print(len(data_dict['labels']))

# data = np.asarray(data_dict['data'])

# labels = np.asarray(data_dict['labels'])

data = data_dict['data']

labels = data_dict['labels']



#print(data)


# label_encoder = preprocessing.LabelEncoder()
# data = set(data)
# data = label_encoder.fit_transform(data)


print(len(data))
print(len(labels))
# print(data,labels)

# x_train = np.array(data)
# y_train = np.array(labels)

# print(data)
print(labels)

li = []

data_temp = []
for i in data:
    if(len(i) == 42):
        data_temp.append(i)
    else:
        data_temp.append(i[:42])

for i in data_temp:
    if(len(i) != 42):
        print("#########################################################")
    print(len(i))

print("data")

# for i in labels:
#     temp=[]
#     temp.append(i)
#     li.append(temp)

# labels = li
# print(labels)
data = data_temp

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()


model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
