import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Veri yükleme
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Veri setinin boyutunu kontrol etme
if len(data) == 0:
    print("Veri seti boş!")
    exit()

#Train ve test'e ayırma
data_padded = pad_sequences(data, padding='post', dtype='float32')     #Sequence to array hatası verirse bu kısım hatayı çözebilir
x_train, x_test, y_train, y_test = train_test_split(data_padded, labels, test_size=0.2, shuffle=True, stratify=labels)  #Sequence to array hatası verirse bu kısım hatayı çözebilir

# Model oluşturma ve eğitme
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Modelin değerlendirilmesi
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly !'.format(score * 100))

# Eğitilen modeli kaydetme
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
