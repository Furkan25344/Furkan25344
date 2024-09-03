import sys
sys.path.append('/yüklü/dizin/yolu')
# Kullanılan kütüphanelerin import edilmesi
import numpy as np # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
import matplotlib.pyplot as plt # type: ignore
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
import torch.optim as optim # type: ignore
from torchvision import datasets, transforms # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
import torch.optim as optim # type: ignore
from torchvision import datasets, transforms # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.metrics import mean_squared_error # type: ignore
from nltk.tokenize import word_tokenize # type: ignore

import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
import torch.optim as optim # type: ignore
from torchvision import datasets, transforms # type: ignore

# Libraries for data manipulation
import pandas as pd # type: ignore
import numpy as np # type: ignore

# Library for data visualization
import matplotlib.pyplot as plt # type: ignore

# Libraries for machine learning
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.metrics import mean_squared_error # type: ignore

# Library for natural language processing
from nltk.tokenize import word_tokenize # type: ignore

import speech_recognition as sr  # type: ignore # Speech recognition library
import pyttsx3 as tts  # type: ignore # Text-to-speech library
import pandas as pd  # type: ignore # Data manipulation library
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore # Text representation
from sklearn.linear_model import LogisticRegression  # type: ignore # Classification model


# Örnek veri oluşturma
np.random.seed(0)
X = np.random.rand(100, 1)
y = 2 + 3 * X + np.random.randn(100, 1)

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Modeli eğitme
model = LinearRegression()
model.fit(X_train, y_train)

# Test verisi üzerinde tahmin yapma
y_pred = model.predict(X_test)

# Sonuçları görselleştirme
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.show()
import tkinter as tk
from tkinter import messagebox
import numpy as np # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
import matplotlib.pyplot as plt # type: ignore

# Basit bir yapay zeka
def yapay_zeka(mesaj):
    if mesaj.lower() == "merhaba":
        return "Merhaba, nasıl yardımcı olabilirim?"
    elif mesaj.lower() == "nasılsın":
        return "Ben bir yapay zeka olduğum için duygularım yok, ama size yardımcı olmaktan mutluluk duyarım."
    else:
        return "Üzgünüm, anlamadım."

# Kullanıcı arayüzü fonksiyonları
def mesaj_gonder():
    mesaj = kullanici_girdisi.get()
    cevap = yapay_zeka(mesaj)
    mesaj_listesi.insert(tk.END, "Sen: " + mesaj)
    mesaj_listesi.insert(tk.END, "Yapay Zeka: " + cevap)
    kullanici_girdisi.delete(0, tk.END)
    messagebox.showinfo("Yapay Zeka Cevap", cevap)

# Kullanıcı arayüzü oluşturma
root = tk.Tk()
root.title("Yapay Zeka İletişim Arayüzü")

frame = tk.Frame(root)
frame.pack(pady=10)

mesaj_listesi = tk.Listbox(frame, width=50, height=20)
mesaj_listesi.pack(side=tk.LEFT, fill=tk.BOTH)

scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL)
scrollbar.config(command=mesaj_listesi.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

kullanici_girdisi = tk.Entry(root, width=50)
kullanici_girdisi.pack(pady=10)

gonder_dugmesi = tk.Button(root, text="Gönder", command=mesaj_gonder)
gonder_dugmesi.pack(pady=10)

# Örnek veri oluşturma
np.random.seed(0)
X = np.random.rand(100, 1)
y = 2 + 3 * X + np.random.randn(100, 1)

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Modeli eğitme
model = LinearRegression()
model.fit(X_train, y_train)

# Test verisi üzerinde tahmin yapma
y_pred = model.predict(X_test)

# Sonuçları görselleştirme
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.show()

root.mainloop()
import subprocess
import sys

# tkinter'ın yüklü olup olmadığını kontrol etme
try:
    subprocess.check_output([sys.executable, '-m', 'tkinter', '-v'])

except subprocess.CalledProcessError:
    # tkinter yüklü değilse yükleme işlemi
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tk'])

import tkinter as tk
from tkinter import messagebox
import numpy as np # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
import matplotlib.pyplot as plt # type: ignore

# Basit bir yapay zeka
def yapay_zeka(mesaj):
    if mesaj.lower() == "merhaba":
        return "Merhaba, nasıl yardımcı olabilirim?"
    elif mesaj.lower() == "nasılsın":
        return "Ben bir yapay zeka olduğum için duygularım yok, ama size yardımcı olmaktan mutluluk duyarım."
    else:
        return "Üzgünüm, anlamadım."

# Kullanıcı arayüzü fonksiyonları
def mesaj_gonder():
    mesaj = kullanici_girdisi.get()
    cevap = yapay_zeka(mesaj)
    mesaj_listesi.insert(tk.END, "Sen: " + mesaj)
    mesaj_listesi.insert(tk.END, "Yapay Zeka: " + cevap)
    kullanici_girdisi.delete(0, tk.END)
    messagebox.showinfo("Yapay Zeka Cevap", cevap)

# Kullanıcı arayüzü oluşturma
root = tk.Tk()
root.title("Yapay Zeka İletişim Arayüzü")

frame = tk.Frame(root)
frame.pack(pady=10)

mesaj_listesi = tk.Listbox(frame, width=50, height=20)
mesaj_listesi.pack(side=tk.LEFT, fill=tk.BOTH)

scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL)
scrollbar.config(command=mesaj_listesi.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

kullanici_girdisi = tk.Entry(root, width=50)
kullanici_girdisi.pack(pady=10)

gonder_dugmesi = tk.Button(root, text="Gönder", command=mesaj_gonder)
gonder_dugmesi.pack(pady=10)

# Örnek veri oluşturma
np.random.seed(0)
X = np.random.rand(100, 1)
y = 2 + 3 * X + np.random.randn(100, 1)

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Modeli eğitme
model = LinearRegression()
model.fit(X_train, y_train)

# Test verisi üzerinde tahmin yapma
y_pred = model.predict(X_test)

# Sonuçları görselleştirme
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.show()

root.mainloop()
from googlesearch import search # type: ignore
import random

def yapay_zeka(mesaj):
    # Google'da arama yapma
    arama_sonuclari = list(search(mesaj, num=3, stop=3, pause=2.0))
    
    # Rastgele bir arama sonucunu seçme
    cevaplar = [
        "Bir araştırma sonucuna göre...",
        "Google'da şunu buldum...",
        "İşte bir cevap..."
    ]
    cevap = random.choice(cevaplar)
    cevap += "\n\n"
    for i, link in enumerate(arama_sonuclari):
        cevap += f"{i+1}. {link}\n"
    return cevap

# Kullanıcı arayüzü fonksiyonları ve tkinter ile arayüz oluşturma işlemleri burada devam eder...
from tkinter import *

def yapay_zeka(mesaj):
    # Burada yapay zekanın cevap üretme işlemleri yapılacak
    cevap = "Yapay zeka cevabı: Bu bir örnek cevaptır."
    return cevap

def soru_sor():
    soru = kullanici_girdisi.get()
    cevap = yapay_zeka(soru)
    mesaj_listesi.insert(END, "Soru: " + soru)
    mesaj_listesi.insert(END, "Cevap: " + cevap)
    kullanici_girdisi.delete(0, END)

root = Tk()
root.title("Yapay Zeka Chat Bot")

frame = Frame(root)
frame.pack(pady=10)

mesaj_listesi = Listbox(frame, width=50, height=20)
mesaj_listesi.pack(side=LEFT, fill=BOTH)

scrollbar = Scrollbar(frame, orient=VERTICAL)
scrollbar.config(command=mesaj_listesi.yview)
scrollbar.pack(side=RIGHT, fill=Y)

kullanici_girdisi = Entry(root, width=50)
kullanici_girdisi.pack(pady=10)

gonder_dugmesi = Button(root, text="Soru Sor", command=soru_sor)
gonder_dugmesi.pack(pady=10)

root.mainloop()
from tkinter import *
from googlesearch import search # type: ignore
import random

# Basit bir yapay zeka
def yapay_zeka(mesaj):
    # Google'da arama yapma
    arama_sonuclari = list(search(mesaj, num=3, stop=3, pause=2.0))
    
    # Rastgele bir arama sonucunu seçme
    cevaplar = [
        "Bir araştırma sonucuna göre...",
        "Google'da şunu buldum...",
        "İşte bir cevap..."
    ]
    cevap = random.choice(cevaplar)
    cevap += "\n\n"
    for i, link in enumerate(arama_sonuclari):
        cevap += f"{i+1}. {link}\n"
    return cevap

def soru_sor():
    soru = kullanici_girdisi.get()
    cevap = yapay_zeka(soru)
    mesaj_listesi.insert(END, "Soru: " + soru)
    mesaj_listesi.insert(END, "Cevap: " + cevap)
    kullanici_girdisi.delete(0, END)

root = Tk()
root.title("Yapay Zeka Chat Bot")

frame = Frame(root)
frame.pack(pady=10)

mesaj_listesi = Listbox(frame, width=50, height=20)
mesaj_listesi.pack(side=LEFT, fill=BOTH)

scrollbar = Scrollbar(frame, orient=VERTICAL)
scrollbar.config(command=mesaj_listesi.yview)
scrollbar.pack(side=RIGHT, fill=Y)

kullanici_girdisi = Entry(root, width=50)
kullanici_girdisi.pack(pady=10)

gonder_dugmesi = Button(root, text="Soru Sor", command=soru_sor)
gonder_dugmesi.pack(pady=10)

root.mainloop()
import pandas as pd # type: ignore

def veri_analizi(veri):
    df = pd.DataFrame(veri)
    return df.describe()
class YapayZeka:
    def __init__(self):
        self.ogrenme_verileri = []

    def geri_bildirim_al(self, feedback):
        self.ogrenme_verileri.append(feedback)

    def kendini_gelistir(self):
        # Basit bir öğrenme süreci örneği
        print("Öğrenme süreci tamamlandı.")

# Yapay zeka sınıfını kullanma örneği
yz = YapayZeka()
yz.geri_bildirim_al("Geri bildirim 1")
yz.geri_bildirim_al("Geri bildirim 2")
yz.kendini_gelistir()
# Gerekli kütüphanelerin import edilmesi
import nltk # type: ignore
from nltk.tokenize import word_tokenize # type: ignore
from nltk.corpus import stopwords # type: ignore
from nltk.stem import WordNetLemmatizer # type: ignore
import tkinter as tk
from tkinter import messagebox
import random
from googlesearch import search # type: ignore

# İlk defa çalıştırıyorsanız bu satırların yorumunu kaldırarak çalıştırın:
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# Metin ön işleme fonksiyonu
def anlamli_kelime_analizi(komut):
    kelimeler = word_tokenize(komut)
    stop_words = set(stopwords.words('turkish'))
    lemmatizer = WordNetLemmatizer()
    anlamli_kelimeler = [lemmatizer.lemmatize(kelime) for kelime in kelimeler if kelime.isalpha() and kelime not in stop_words]
    return anlamli_kelimeler

# Basit Yapay Zeka sınıfı
class YapayZeka:
    def __init__(self):
        self.ogrenme_verileri = []

    def geri_bildirim_al(self, feedback):
        self.ogrenme_verileri.append(feedback)

    def kendini_gelistir(self):
        # Basit bir öğrenme süreci örneği
        print("Öğrenme süreci tamamlandı.")

    def google_arama(self, mesaj):
        # Google'da arama yapma
        arama_sonuclari = list(search(mesaj, num=3, stop=3, pause=2.0))
        
        # Rastgele bir arama sonucunu seçme
        cevaplar = [
            "Bir araştırma sonucuna göre...",
            "Google'da şunu buldum...",
            "İşte bir cevap..."
        ]
        cevap = random.choice(cevaplar)
        cevap += "\n\n"
        for i, link in enumerate(arama_sonuclari):
            cevap += f"{i+1}. {link}\n"
        return cevap

    def cevap_uret(self, mesaj):
        anlamli_kelimeler = anlamli_kelime_analizi(mesaj)
        if not anlamli_kelimeler:
            return "Anlamadım, lütfen tekrarlar mısınız?"

        # Basit kurallar ile cevap üretme
        if "merhaba" in anlamli_kelimeler:
            return "Merhaba, nasıl yardımcı olabilirim?"
        elif "nasılsın" in anlamli_kelimeler:
            return "Ben bir yapay zeka olduğum için duygularım yok, ama size yardımcı olmaktan mutluluk duyarım."
        elif "arama" in anlamli_kelimeler:
            return self.google_arama(mesaj)
        else:
            return "Üzgünüm, anlamadım."

# Kullanıcı arayüzü fonksiyonları
def mesaj_gonder():
    mesaj = kullanici_girdisi.get()
    cevap = yapay_zeka_instance.cevap_uret(mesaj)
    mesaj_listesi.insert(tk.END, "Sen: " + mesaj)
    mesaj_listesi.insert(tk.END, "Yapay Zeka: " + cevap)
    kullanici_girdisi.delete(0, tk.END)
    messagebox.showinfo("Yapay Zeka Cevap", cevap)

# Kullanıcı arayüzü oluşturma
root = tk.Tk()
root.title("Yapay Zeka İletişim Arayüzü")

frame = tk.Frame(root)
frame.pack(pady=10)

mesaj_listesi = tk.Listbox(frame, width=50, height=20)
mesaj_listesi.pack(side=tk.LEFT, fill=tk.BOTH)

scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL)
scrollbar.config(command=mesaj_listesi.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

mesaj_listesi.config(yscrollcommand=scrollbar.set)

kullanici_girdisi = tk.Entry(root, width=50)
kullanici_girdisi.pack(pady=10)

gonder_dugmesi = tk.Button(root, text="Gönder", command=mesaj_gonder)
gonder_dugmesi.pack(pady=10)

# Yapay Zeka örneği oluşturma
yapay_zeka_instance = YapayZeka()

root.mainloop()
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
import torch.optim as optim # type: ignore
from torchvision import datasets, transforms # type: ignore
import nltk # type: ignore
from nltk.tokenize import word_tokenize # type: ignore

# İlk defa çalıştırıyorsanız bu satırın yorumunu kaldırarak çalıştırın:
# nltk.download('punkt')

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def train(model, train_loader, optimizer, criterion, epochs=5):
    model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            data = data.view(data.shape[0], -1)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

def get_data_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root='./data', download=True, train=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    return loader

class YapayZeka:
    def __init__(self):
        self.ogrenme_verileri = []

    def geri_bildirim_al(self, feedback):
        self.ogrenme_verileri.append(feedback)

    def kendini_gelistir(self):
        # Basit bir öğrenme süreci örneği
        print("Öğrenme süreci tamamlandı.")

    def anlamli_kelime_analizi(self, komut):
        kelimeler = word_tokenize(komut)
        anlamli_kelimeler = [kelime for kelime in kelimeler if kelime.isalpha()]
        return anlamli_kelimeler

    def komut_isle(self, komut):
        anlamli_kelimeler = self.anlamli_kelime_analizi(komut)
        if 'eğit' in anlamli_kelimeler:
            print("Model eğitiliyor...")
            model = MNISTModel()
            loader = get_data_loader()
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            criterion = nn.CrossEntropyLoss()
            train(model, loader, optimizer, criterion)
        elif 'öğren' in anlamli_kelimeler:
            self.kendini_gelistir()
        else:
            print("Üzgünüm, komutu anlayamadım.")

if __name__ == "__main__":
    yapay_zeka = YapayZeka()
    komut = input("Komut girin: ")
    yapay_zeka.komut_isle(komut)
# Kullanılan kütüphanelerin import edilmesi
import numpy as np # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
import matplotlib.pyplot as plt # type: ignore
import tkinter as tk
from tkinter import messagebox
from googlesearch import search # type: ignore
import random

# Örnek veri oluşturma
np.random.seed(0)
X = np.random.rand(100, 1)
y = 2 + 3 * X + np.random.randn(100, 1)

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Modeli eğitme
model = LinearRegression()
model.fit(X_train, y_train)

# Test verisi üzerinde tahmin yapma
y_pred = model.predict(X_test)

# Sonuçları görselleştirme
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.show()

# Basit bir yapay zeka
def yapay_zeka(mesaj):
    if mesaj.lower() == "merhaba":
        return "Merhaba, nasıl yardımcı olabilirim?"
    elif mesaj.lower() == "nasılsın":
        return "Ben bir yapay zeka olduğum için duygularım yok, ama size yardımcı olmaktan mutluluk duyarım."
    else:
        return "Üzgünüm, anlamadım."

# Kullanıcı arayüzü fonksiyonları
def mesaj_gonder():
    mesaj = kullanici_girdisi.get()
    cevap = yapay_zeka(mesaj)
    mesaj_listesi.insert(tk.END, "Sen: " + mesaj)
    mesaj_listesi.insert(tk.END, "Yapay Zeka: " + cevap)
    kullanici_girdisi.delete(0, tk.END)
    messagebox.showinfo("Yapay Zeka Cevap", cevap)

# Kullanıcı arayüzü oluşturma
root = tk.Tk()
root.title("Yapay Zeka İletişim Arayüzü")

frame = tk.Frame(root)
frame.pack(pady=10)

mesaj_listesi = tk.Listbox(frame, width=50, height=20)
mesaj_listesi.pack(side=tk.LEFT, fill=tk.BOTH)

scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL)
scrollbar.config(command=mesaj_listesi.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

kullanici_girdisi = tk.Entry(root, width=50)
kullanici_girdisi.pack(pady=10)

gonder_dugmesi = tk.Button(root, text="Gönder", command=mesaj_gonder)
gonder_dugmesi.pack(pady=10)

root.mainloop()
import pyttsx3 # type: ignore
import tkinter as tk
import nltk # type: ignore
from nltk.tokenize import word_tokenize # type: ignore

# Sesi başlat
engine = pyttsx3.init()

# Sesli iletişim için fonksiyon
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Grafik arayüzü için fonksiyonlar
def button_click():
    user_input = entry.get()
    response = handle_command(user_input)
    label.config(text=response)
    speak(response)

def handle_command(command):
    if command == "merhaba":
        return "Merhaba, nasıl yardımcı olabilirim?"
    elif command == "selam":
        return "Selam, ben buradayım!"
    else:
        return "Üzgünüm, anlamadım."

# Doğal Dil İşleme (NLP) fonksiyonları
def analyze_text(text):
    tokens = word_tokenize(text)
    return tokens

# GUI oluşturma
root = tk.Tk()
root.title("Regression Bot")

entry = tk.Entry(root, width=50)
entry.pack()

button = tk.Button(root, text="Gönder", command=button_click)
button.pack()

label = tk.Label(root, text="")
label.pack()

root.mainloop()
import tensorflow as tf # type: ignore
import torch # type: ignore

# TensorFlow kullanarak basit bir yapay sinir ağı oluşturma
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

# PyTorch kullanarak basit bir yapay sinir ağı oluşturma
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(784, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.dropout(x, p=0.2, training=self.training)
        x = self.fc2(x)
        return x

model = Net()

import os
import sys
import time
import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.metrics import mean_squared_error # type: ignore
from sklearn.metrics import r2_score # type: ignore
from googlesearch import search # type: ignore
from pyttsx3 import init # type: ignore

# Örnek bir veri kümesi oluşturalım
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Veri kümesini eğitim ve test setlerine bölelim
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Doğrusal regresyon modelini oluşturalım
model = LinearRegression()
model.fit(X_train, y_train)

# Test seti üzerinde modeli değerlendirelim
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Sonuçları ekrana yazdıralım
print("MSE:", mse)
print("R2 Score:", r2)

# Google üzerinde arama yapalım
query = "Python Artificial Intelligence"
for j in search(query, num=5, stop=5, pause=2):
    print(j)

# Sesli geri bildirim sağlayalım
engine = init()
engine.say("Regression analysis completed successfully.")
engine.runAndWait()
import googlesearch # type: ignore
import pyttsx3 # type: ignore

def main():
    try:
        # Buraya mevcut kodunuzu ekleyin
        pass
    except Exception as e:
        print("Hata oluştu:", e)

if __name__ == "__main__":
    main()
# Dosyayı açın
with open('dosya.txt', 'r') as dosya:
    # Her satırı okuyun
    for satir in dosya:
        # Her satırı ekrana yazdırın
        print(satir)

from keras.models import Sequential # type: ignore
from keras.layers import Dense, Dropout # type: ignore
import numpy as np # type: ignore

# Örnek veri oluşturma
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Model oluşturma
model = Sequential()
model.add(Dense(8, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Modeli derleme
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Modeli eğitme
model.fit(X, y, epochs=1000, batch_size=4)
while True:
    # Kullanıcıdan soru al
    input_data = input("Soru: ")

    # Modelin cevap vermesi
    input_vector = np.array([[int(x) for x in input_data.split()]])
    prediction = model.predict(input_vector)

    # Cevabı ekrana yazdır
    print("Cevap:", prediction[0][0])
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Örnek veri oluşturma
X = tf.constant([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=tf.float32)
y = tf.constant([[0], [1], [1], [0]], dtype=tf.float32)

# Model oluşturma
model = Sequential([
    Dense(8, input_dim=2, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# Modeli derleme
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Modeli eğitme
model.fit(X, y, epochs=1000, batch_size=4, verbose=0)

while True:
    # Kullanıcıdan soru al
    input_data = input("Soru: ")

    # Modelin cevap vermesi
    input_vector = tf.constant([[int(x) for x in input_data.split()]], dtype=tf.float32)
    prediction = model.predict(input_vector)

    # Cevabı ekrana yazdır
    print("Cevap:", prediction[0][0])
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import tensorflow as tf
from tensorflow.keras import layers, models
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import time

def search_engine(query, search_engine_url, next_page_xpath, result_class):
    # WebDriver'ı başlat
    driver_path = "your_driver_path"
    driver = webdriver.Chrome(executable_path=driver_path)

    # Arama motoruna bağlan
    driver.get(search_engine_url)

    # Arama kutusuna sorguyu yaz ve ara
    search_box = driver.find_element_by_name("q")
    search_box.send_keys(query)
    search_box.send_keys(Keys.RETURN)

    # Sayfa sayısı belirle
    page_count = 3  # Örnek olarak ilk 3 sayfayı tarayacak şekilde ayarlandı

    # Tüm sayfaları dolaş
    for i in range(page_count):
        # Sayfanın HTML içeriğini al
        html = driver.page_source

        # BeautifulSoup ile HTML'i analiz et
        soup = BeautifulSoup(html, "html.parser")

        # İlgili verileri çekme
        results = soup.find_all("div", class_=result_class)
        for result in results:
            try:
                title = result.find("h3").get_text()
                link = result.find("a")["href"]
                print(f"Title: {title}\nLink: {link}\n")
            except AttributeError:
                pass

        # Sayfa numarasını güncelle
        page_num = i + 2
        # Sonraki sayfaya git
        next_page = driver.find_element_by_xpath(next_page_xpath)
        next_page.click()

        # Sayfayı biraz beklet
        time.sleep(2)

    # WebDriver'ı kapat
    driver.quit()

# Örnek kullanım
search_query = "your_search_query_here"
search_engine_url = "https://www.bing.com"
next_page_xpath = "//a[@class='sb_pagN']"
result_class = "b_algo"
search_engine(search_query, search_engine_url, next_page_xpath, result_class)
import requests
import json

# JSON dosyasını indir
url = "https://googlechromelabs.github.io/chrome-for-testing/known-good-versions-with-downloads.json"
response = requests.get(url)
data = response.json()

# Aranacak sürüm
target_version = "126.0.6478.115"

# Sürüm için ChromeDriver bağlantısını bul
for version in data['versions']:
    if version['version'] == target_version:
        downloads = version['downloads']
        for download in downloads['chromedriver']:
            if download['platform'] == 'win32':  # Windows platformu için
                download_url = download['url']
                print(f"Download URL for ChromeDriver {target_version}: {download_url}")

# İndirilecek URL'yi yazdır
import requests

# Yukarıdaki koddan elde edilen indirme URL'si
download_url = "https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/126.0.6478.115/win32/chromedriver-win32.zip"
response = requests.get(download_url)

# İndirilen dosyayı kaydet
with open("chromedriver-win32.zip", "wb") as file:
    file.write(response.content)


import requests
import json

# JSON dosyasını indir
url = "https://googlechromelabs.github.io/chrome-for-testing/known-good-versions-with-downloads.json"
response = requests.get(url)
data = response.json()

# Aranacak sürüm
target_version = "126.0.6478.115"

# Sürüm için ChromeDriver bağlantısını bul
for version in data['versions']:
    if version['version'] == target_version:
        downloads = version['downloads']
        for download in downloads['chromedriver']:
            if download['platform'] == 'win32':  # Windows platformu için
                download_url = download['url']
                print(f"Download URL for ChromeDriver {target_version}: {download_url}")
import requests
import json

# JSON dosyasını indir
url = "https://googlechromelabs.github.io/chrome-for-testing/known-good-versions-with-downloads.json"
response = requests.get(url)
data = response.json()

# Aranacak sürüm
target_version = "126.0.6478.115"

# Sürüm için ChromeDriver bağlantısını bul
for version in data['versions']:
    if version['version'] == target_version:
        downloads = version['downloads']
        for download in downloads['chromedriver']:
            if download['platform'] == 'win32':  # Windows platformu için
                download_url = download['url']
                print(f"Download URL for ChromeDriver {target_version}: {download_url}")
import requests
import zipfile
import os

# Yukarıdaki koddan elde edilen indirme URL'si
download_url = "https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/126.0.6478.115/win32/chromedriver-win32.zip"

# İndirilen dosyanın kaydedileceği yol
zip_file_path = "C:\\Users\\KullanıcıAdı\\Downloads\\chromedriver-win32.zip"

# İndirilen dosyayı kaydet
response = requests.get(download_url)
with open(zip_file_path, "wb") as file:
    file.write(response.content)

# Çıkarılacak klasör yolu
extract_folder_path = "C:\\Users\\KullanıcıAdı\\Downloads\\chromedriver"

# Zip dosyasını çıkar
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder_path)

# Çıkarılan dosyaların kontrolü
for root, dirs, files in os.walk(extract_folder_path):
    for file in files:
        print(os.path.join(root, file))
import requests
import json

# JSON dosyasını indir
url = "https://googlechromelabs.github.io/chrome-for-testing/known-good-versions-with-downloads.json"
response = requests.get(url)
data = response.json()

# Aranacak sürüm
target_version = "126.0.6478.115"

# Sürüm için ChromeDriver bağlantısını bul
download_url = ""
for version in data['versions']:
    if version['version'] == target_version:
        downloads = version['downloads']
        for download in downloads['chromedriver']:
            if download['platform'] == 'win32':  # Windows platformu için
                download_url = download['url']
                print(f"Download URL for ChromeDriver {target_version}: {download_url}")
                break
import requests
import zipfile
import os

# İndirilecek dosyanın URL'si (yukarıdaki koddan elde edilen URL'yi kullanın)
download_url = "https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/126.0.6478.115/win32/chromedriver-win32.zip"

# İndirilen dosyanın kaydedileceği yol
desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
zip_file_path = os.path.join(desktop_path, "chromedriver-win32.zip")

# Dosyayı indirin ve kaydedin
response = requests.get(download_url)
with open(zip_file_path, "wb") as file:
    file.write(response.content)

# Çıkarılacak klasör yolu
extract_folder_path = os.path.join(desktop_path, "chromedriver")

# Zip dosyasını çıkar
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder_path)

# Çıkarılan dosyaların kontrolü
for root, dirs, files in os.walk(extract_folder_path):
    for file in files:
        print(os.path.join(root, file))
from selenium import webdriver # type: ignore
from selenium.webdriver.common.keys import Keys # type: ignore
from bs4 import BeautifulSoup
import time

def search_engine(query, search_engine_url, next_page_xpath, result_class):
    driver_path = os.path.join(desktop_path, "chromedriver", "chromedriver.exe")  # WebDriver'ın yüklü olduğu yol
    driver = webdriver.Chrome(executable_path=driver_path)

    driver.get(search_engine_url)

    search_box = driver.find_element_by_name("q")
    search_box.send_keys(query)
    search_box.send_keys(Keys.RETURN)

    page_count = 3

    for i in range(page_count):
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")

        results = soup.find_all("div", class_=result_class)
        for result in results:
            try:
                title = result.find("h3").get_text()
                link = result.find("a")["href"]
                print(f"Title: {title}\nLink: {link}\n")
            except AttributeError:
                pass

        try:
            next_page = driver.find_element_by_xpath(next_page_xpath)
            next_page.click()
        except:
            print("Sonraki sayfa bulunamadı veya başka sayfa yok.")
            break

        time.sleep(2)

    driver.quit()

search_query = "yapay zeka"

# Google
search_engine(search_query, "https://www.google.com", "//a[@id='pnnext']", "g")

# Bing
search_engine(search_query, "https://www.bing.com", "//a[@class='sb_pagN']", "b_algo")

# DuckDuckGo
search_engine(search_query, "https://duckduckgo.com", "//a[@class='result--more__btn']", "result__title")
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Model ve tokenizer'ı yükleme
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Sohbet başlatma
print("Yapay Zeka: Merhaba, size nasıl yardımcı olabilirim? Çıkmak için 'exit' yazabilirsiniz.")

while True:
    # Kullanıcıdan giriş al
    user_input = input("Siz: ")

    # Çıkış kontrolü
    if user_input.lower() == 'exit':
        print("Yapay Zeka: Güle güle!")
        break

    # Tokenize etme
    input_ids = tokenizer.encode(user_input, return_tensors='pt')

    # Yapay zekaya cevap ürettirme
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=model.config.pad_token_id)

    # Cevabı ekrana yazdırma
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Yapay Zeka:", response)
import nltk
from nltk.chat.util import Chat
import random
import datetime
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from snowballstemmer import TurkishStemmer

# Gereken veri setlerini indir
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    """
    Metni önişleme fonksiyonu:
    - Küçük harfe çevirir
    - Noktalama işaretlerini kaldırır
    - Gereksiz kelimeleri (stop words) kaldırır
    - Kelime köklerini bulur (stemming) - Türkçe için TurkishStemmer kullanarak
    """
    text = text.lower()
    words = word_tokenize(text)
    words = [word for word in words if word.isalnum()]
    words = [word for word in words if word not in stopwords.words('turkish')]
    stemmer = TurkishStemmer()
    words = [stemmer.stemWord(word) for word in words]
    return words

def get_pairs():
    """
    Chatbot'un yanıt vereceği kalıpları tanımlayan fonksiyon.
    """
    return [
        [
            r"merhaba|selam|hey",
            ["Merhaba!", "Selam!", "Hey!"]
        ],
        [
            r"nasılsın?",
            ["İyiyim, teşekkürler. Sen nasılsın?", "Harikayım, ya sen?"]
        ],
        [
            r"ben de iyiyim|ben de iyiyim, teşekkürler",
            ["Bunu duyduğuma sevindim!", "Harika!"]
        ],
        [
            r"adın ne?",
            ["Ben bir yapay zeka botuyum.", "Benim adım ChatBot."]
        ],
        [
            r"kaç yaşındasın?",
            ["Yaşım yok, çünkü ben bir yapay zeka programıyım.", "Ben zamanın ötesindeyim!"]
        ],
        [
            r"neler yapabilirsin?",
            ["Sana birçok konuda yardımcı olabilirim. Örneğin, sorularını yanıtlayabilir, bilgi sağlayabilir, sohbet edebilir veya fıkra anlatabilirim."]
        ],
        [
            r"fıkra anlatır mısın?",
            [
                "Neden programcılar kahve içer?\nÇünkü kodları derlemek için!",
                "Bir bilgisayar neden doktora gitmiş?\nVirüs kaptığı için!"
            ]
        ],
        [
            r"bugünün tarihi ne?",
            ["Bugünün tarihi: " + datetime.datetime.now().strftime("%d/%m/%Y")]
        ],
        [
            r"saat kaç?",
            ["Saat şu anda: " + datetime.datetime.now().strftime("%H:%M:%S")]
        ],
        [
            r"görüşürüz|hoşçakal|çık|exit",
            ["Görüşmek üzere!", "Hoşçakal!"]
        ],
    ]

# Türkçe yansımalar (özelleştirilmiş)
my_reflections = {
    "ben"  : "sen",
    "sen"  : "ben",
    "beni" : "seni",
    "sana" : "bana",
    "benim": "senin",
    "senin": "benim"
}

def handle_user_input(user_input, chatbot):
    """
    Kullanıcı girdisini işleyip chatbot yanıtı döndüren fonksiyon.
    """
    # Yansımaları uygula
    for key, value in my_reflections.items():
        user_input = user_input.replace(key, value)

    user_input_processed = preprocess_text(user_input)

    # Eşleşen bir kalıp yoksa en uygun yanıtı bulmaya çalış
    best_match = None
    best_match_score = 0
    for pattern, responses in get_pairs():
        processed_pattern = preprocess_text(pattern)
        score = sum(word in user_input_processed for word in processed_pattern)
        if score > best_match_score:
            best_match = responses
            best_match_score = score

    if best_match:
        return random.choice(best_match)
    else:
        return random.choice(["Bunu anlamadım, tekrar eder misin?", "Üzgünüm, bu konuda bilgim yok."])

def start_chatbot():
    """
    Chatbot'u başlatan ana fonksiyon.
    """
    print("Merhaba! Benimle konuşabilirsin. Çıkmak için 'görüşürüz', 'hoşçakal', 'çık' veya 'exit' yazabilirsin.")
    chatbot = Chat(get_pairs())

    while True:
        user_input = input("> ")
        if user_input.lower() in ["görüşürüz", "hoşçakal", "çık", "exit"]:
            print("Görüşmek üzere!")
            break

        response = handle_user_input(user_input, chatbot)
        print(response)

if __name__ == "__main__":
    start_chatbot()
import tkinter as tk

def on_click():
    label.config(text="Merhaba, {}!".format(entry.get()))

root = tk.Tk()
root.title("Hoş Geldiniz")

frame = tk.Frame(root)
frame.pack(padx=20, pady=20)

label = tk.Label(frame, text="Adınızı Girin:")
label.pack()

entry = tk.Entry(frame)
entry.pack()

button = tk.Button(frame, text="Gönder", command=on_click)
button.pack()

root.mainloop()
def listen_to_user():
    # Initialize speech recognizer
    r = sr.Recognizer()

    # Set up microphone
    with sr.Microphone() as source:
        print("Listening...")

        # Listen for speech input
        audio = r.listen(source)

        try:
            # Convert speech to text
            text = r.recognize_google(audio)
            print("You said: {}".format(text))
            return text
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
            return None

def speak_to_user(text):
    # Initialize speech engine
    engine = tts.init()

    # Set up voice properties
    engine.setProperty('rate', 150)
    engine.setProperty('voice', engine.getProperty('voices')[0].id)

    # Convert text to speech and play
    engine.say(text)
    engine.runAndWait()

    # Get user input as text
    user_input = listen_to_user()

    # Convert user input to lowercase and tokenize
    input_tokens = word_tokenize(user_input.lower())

    # Use a trained classification model to predict intent
    intent = classifier.predict([input_tokens])[0]

    # Respond based on the predicted intent
    if intent == "greet":
        speak_to_user("Hello there! How can I help you today?")
    elif intent == "ask_question":
        # Process and answer the question using appropriate methods
        answer = process_question(user_input) # type: ignore
        speak_to_user(answer)
    elif intent == "give_command":
        # Execute the given command
        execute_command(user_input) # type: ignore
        speak_to_user("Command executed.")
    else:
        speak_to_user("Sorry, I didn't understand what you said.")

# Train a classification model using labeled data ( intents and corresponding text representations)
classifier = LogisticRegression()
# Load trained model or train it using labeled data
def handle_user_interaction():
    # Get user input as text
    user_input = listen_to_user()

    # Convert user input to lowercase and tokenize
    input_tokens = word_tokenize(user_input.lower())

    # Use a trained classification model to predict intent
    intent = classifier.predict([input_tokens])[0]

    # Respond based on the predicted intent
    if intent == "greet":
        speak_to_user("Hello there! How can I help you today?")
    elif intent == "ask_question":
        # Process and answer the question using appropriate methods
        answer = process_question(user_input) # type: ignore
        speak_to_user(answer)
    elif intent == "give_command":
        # Execute the given command
        execute_command(user_input) # type: ignore
        speak_to_user("Command executed.")
    else:
        speak_to_user("Sorry, I didn't understand what you said.")

# Train a classification model using labeled data ( intents and corresponding text representations)
classifier = LogisticRegression()
# Load trained model or train it using labeled data
def process_question(question):
    # Use NLP techniques to extract relevant information from the question
    # Access and process knowledge base or external resources to answer the question
    # Generate a comprehensive and informative answer

    return "Your question is being processed. Please wait."

def process_question(question):
    # Use NLP techniques to extract relevant information from the question
    # Access and process knowledge base or external resources to answer the question
    # Generate a comprehensive and informative answer

    return "Your question is being processed. Please wait."
# Örnek veri oluşturma
np.random.seed(0)
X = np.random.rand(100, 1)
y = 2 + 3 * X + np.random.randn(100, 1)

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Modeli eğitme
model = LinearRegression()
model.fit(X_train, y_train)

# Test verisi üzerinde tahmin yapma
y_pred = model.predict(X_test)

# Sonuçları görselleştirme
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def yapay_zeka(mesaj):
    if mesaj.lower() == "merhaba":
        return "Merhaba, nasıl yardımcı olabilirim?"
    elif mesaj.lower() == "nasılsın":
        return "Ben bir yapay zeka olduğum için duygularım yok, ama size yardımcı olmaktan mutluluk duyarım."
    else:
        return "Üzgünüm, anlamadım."

def mesaj_gonder():
    mesaj = kullanici_girdisi.get()
    cevap = yapay_zeka(mesaj)
    mesaj_listesi.insert(tk.END, "Sen: " + mesaj)
    mesaj_listesi.insert(tk.END, "Yapay Zeka: " + cevap)
    kullanici_girdisi.delete(0, tk.END)
    messagebox.showinfo("Yapay Zeka Cevap", cevap)

root = tk.Tk()
root.title("Yapay Zeka İletişim Arayüzü")

frame = tk.Frame(root)
frame.pack(pady=10)

mesaj_listesi = tk.Listbox(frame, width=50, height=20)
mesaj_listesi.pack(side=tk.LEFT, fill=tk.BOTH)

scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL)
scrollbar.config(command=mesaj_listesi.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

mesaj_listesi.config(yscrollcommand=scrollbar.set)

kullanici_girdisi = tk.Entry(root, width=50)
kullanici_girdisi.pack(pady=10)

gonder_dugmesi = tk.Button(root, text="Gönder", command=mesaj_gonder)
gonder_dugmesi.pack(pady=10)

root.mainloop()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Örnek veri oluşturma
np.random.seed(0)
X = np.random.rand(100, 1)
y = 2 + 3 * X + np.random.randn(100, 1)

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Modeli eğitme
model = LinearRegression()
model.fit(X_train, y_train)

# Test verisi üzerinde tahmin yapma
y_pred = model.predict(X_test)

# Sonuçları görselleştirme
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.show()

import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import tkinter as tk
from tkinter import messagebox

# Örnek veri oluşturma
np.random.seed(0)
X = np.random.rand(100, 1)
y = 2 + 3 * X + np.random.randn(100, 1)

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Modeli eğitme
model = LinearRegression()
model.fit(X_train, y_train)

# Test verisi üzerinde tahmin yapma
y_pred = model.predict(X_test)

# Sonuçları görselleştirme
def plot_regression():
    plt.scatter(X_test, y_test, color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.show()

# Basit yapay zeka işlevi
def yapay_zeka(mesaj):
    if mesaj.lower() == "merhaba":
        return "Merhaba, nasıl yardımcı olabilirim?"
    elif mesaj.lower() == "nasılsın":
        return "Ben bir yapay zeka olduğum için duygularım yok, ama size yardımcı olmaktan mutluluk duyarım."
    else:
        return "Üzgünüm, anlamadım."

# Kullanıcı arayüzü fonksiyonları
def mesaj_gonder():
    mesaj = kullanici_girdisi.get()
    cevap = yapay_zeka(mesaj)
    mesaj_listesi.insert(tk.END, "Sen: " + mesaj)
    mesaj_listesi.insert(tk.END, "Yapay Zeka: " + cevap)
    kullanici_girdisi.delete(0, tk.END)
    messagebox.showinfo("Yapay Zeka Cevap", cevap)

# Kullanıcı arayüzü oluşturma
root = tk.Tk()
root.title("Yapay Zeka İletişim Arayüzü")

frame = tk.Frame(root)
frame.pack(pady=10)

mesaj_listesi = tk.Listbox(frame, width=50, height=20)
mesaj_listesi.pack(side=tk.LEFT, fill=tk.BOTH)

scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL)
scrollbar.config(command=mesaj_listesi.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

mesaj_listesi.config(yscrollcommand=scrollbar.set)

kullanici_girdisi = tk.Entry(root, width=50)
kullanici_girdisi.pack(pady=10)

gonder_dugmesi = tk.Button(root, text="Gönder", command=mesaj_gonder)
gonder_dugmesi.pack(pady=10)

# Grafik çizimi butonu
plot_button = tk.Button(root, text="Regresyon Grafiğini Göster", command=plot_regression)
plot_button.pack(pady=10)

root.mainloop()

