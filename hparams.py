import re, io, copy, shutil, cv2, os, editdistance
from os.path import join
import numpy as np
from collections import Counter
from tqdm import tqdm

class Hparams():
    def __init__(self):
        # Путь до чекпоинта
        self.chk = ''
        
        #В этой папке лежат txt файлы перевода
        self.trans_dir = 'train/words'
        
        #В этой папке лежат  jpg файлы изображений
        self.image_dir = 'train/images'
        
        self.test_dir = '/data/'
        
        #Символы, которые надо удалить
        self.del_sym = ['b', 'd', 'a', 'c', '×', '⊕', ')', '|', 'n', 'm', 'g', 'ǂ', '/', 'k', 'o', '–', '⊗', 'l', '…', 'u','h','і', 'f','t','p', 'r', 'e','s']
        
        # Скорость обучения
        self.lr = 1e-4
        
        # Размер батча
        self.batch_size = 16 
        
        # Размер скрытого слоя
        self.hidden = 512
        
        # Слоёв кодировщика
        self.enc_layers = 1
        
        # Слоёв декодера
        self.dec_layers = 1
        
        # голов внимания
        self.nhead = 4
        
        # Дропаут
        self.dropout = 0.1
 
        # размеры изображения 
        self.width = 1024
        self.height = 128
        
# Загружаем гиперпараметры
hp = Hparams()

# функция игнорируются примеры, содержащие del_sym."""
def process_texts(image_dir,trans_dir):
    lens,lines,names = [],[],[]
    letters = ''
    all_word = {}
    all_files = os.listdir(trans_dir)
    for filename in os.listdir(image_dir):
        if filename[:-3]+'txt' in all_files:
            name, _ = os.path.splitext(filename)
            txt_filepath = join(trans_dir, name + '.txt')
            with open(txt_filepath, 'r', encoding="utf-8") as file:
                data = file.read()
                if len(data)==0:
                    continue
                if len(set(data).intersection(hp.del_sym))>0:
                    continue
                lines.append(data)
                names.append(filename)
                lens.append(len(data))
                letters = letters + ' ' + data
    words = letters.split()
    for word in words:
        if not word in all_word:
            all_word[word] = 0
        else:
            all_word[word] += 1
    
    del_cnt = []
    cnt = Counter(letters)
    """
    #print(cnt)
    for i in cnt:
      if cnt[i]<11:
        del_cnt.append(i)
    #print(del_cnt)  
    """
    print('Максимальная длина строки:', max(Counter(lens).keys()))
    return names,lines,cnt,all_word

# Перевести текст в массив интдексов
def text_to_labels(s, p2idx):
    return [p2idx['SOS']] + [p2idx[i] for i in s if i in p2idx.keys()] + [p2idx['EOS']]

# Перевести индексы в текст
def labels_to_text(s, idx2p):
    S = "".join([idx2p[i] for i in s])
    if S.find('EOS') == -1:
        return S
    else:
        return S[:S.find('EOS')]

# Подсчитать CER
def phoneme_error_rate(p_seq1, p_seq2):
    p_vocab = set(p_seq1 + p_seq2)
    p2c = dict(zip(p_vocab, range(len(p_vocab))))
    c_seq1 = [chr(p2c[p]) for p in p_seq1]
    c_seq2 = [chr(p2c[p]) for p in p_seq2]
    return editdistance.eval(''.join(c_seq1),
                                ''.join(c_seq2)) / len(c_seq2)

# Следующая функция подгружает изображения, меняет их до необходимого размера и нормирует."""
def process_image(img):
    img  = np.stack([img, img, img], axis=-1)
    w, h,_ = img.shape
    
    new_w = hp.height
    new_h = int(h * (new_w / w)) 
    img = cv2.resize(img, (new_h, new_w))
    w, h,_ = img.shape
    
    img = img.astype('float32')
    
    new_h = hp.width
    if h < new_h:
        add_zeros = np.full((w, new_h-h,3), 255)
        img = np.concatenate((img, add_zeros), axis=1)
    
    if h > new_h:
        img = cv2.resize(img, (new_h,new_w))
    
    return img

def generate_data(names,image_dir='train1/images'):
    data_images = []
    for name in tqdm(names):
        img = cv2.imread(image_dir+'/'+name,cv2.IMREAD_GRAYSCALE)#
        img = process_image(img)
        data_images.append(img.astype('uint8'))
    return data_images

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)    