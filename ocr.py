import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
#from tqdm.notebook import tqdm
from hparams import Hparams, process_texts, text_to_labels, labels_to_text, phoneme_error_rate, process_image, generate_data, count_parameters
import cv2, os, argparse, time, random, math
from torchvision import transforms, models
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Сделать текст в батче одной длины
class TextCollate():
    def __call__(self, batch):
        x_padded = []
        max_y_len = max([i[1].size(0) for i in batch])
        y_padded = torch.LongTensor(max_y_len, len(batch))
        y_padded.zero_()

        for i in range(len(batch)):
            x_padded.append(batch[i][0].unsqueeze(0))
            y = batch[i][1]
            y_padded[:y.size(0), i] = y
        
        x_padded = torch.cat(x_padded)
        return x_padded, y_padded   


# Датасет загрузки изображений и тексты
class TextLoader(torch.utils.data.Dataset):
    def __init__(self,name_image,label,image_dir,eval=False):
        self.name_image = name_image
        self.label = label
        self.image_dir = image_dir
        self.eval = eval
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((int(hp.height*1.05), int(hp.width*1.05))),
            transforms.RandomCrop((hp.height, hp.width)),
            transforms.RandomRotation(degrees=(-2, 2),fill=255),
            transforms.ToTensor()
            ])
     
    def __getitem__(self, index):
        img = self.name_image[index]
        if not self.eval:
            img = self.transform(img)
            img = img / img.max()
            img = img**(random.random()*0.7 + 0.6)
        else:
            img = np.transpose(img,(2,0,1))
            img = img / img.max()

        label = text_to_labels(self.label[index], p2idx)
        return (torch.FloatTensor(img), torch.LongTensor(label))

    def __len__(self):
        return len(self.label)


# Кодирование позиции символа
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x)


# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, name, outtoken, hidden = 128, enc_layers=1, dec_layers=1, nhead = 1, dropout=0.1,pretrained=False):
        super(TransformerModel, self).__init__()
        self.backbone = models.__getattribute__(name)(pretrained=pretrained)
        #self.backbone.avgpool = nn.MaxPool2d((4, 1))
        self.backbone.fc = nn.Conv2d(2048, hidden//4, 1)
        
        self.pos_encoder = PositionalEncoding(hidden, dropout)
        self.decoder = nn.Embedding(outtoken, hidden)
        self.pos_decoder = PositionalEncoding(hidden, dropout)
        self.transformer = nn.Transformer(d_model=hidden, nhead=nhead, num_encoder_layers=enc_layers, num_decoder_layers=dec_layers, dim_feedforward=hidden*4, dropout=dropout, activation='relu')
        
        self.fc_out = nn.Linear(hidden, outtoken)
        self.src_mask = None
        self.trg_mask = None
        self.memory_mask = None

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask==1, float('-inf'))
        return mask

    def make_len_mask(self, inp):
        return (inp == 0).transpose(0, 1)

    def forward(self, src, trg):
        if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
            self.trg_mask = self.generate_square_subsequent_mask(len(trg)).to(trg.device)

        x = self.backbone.conv1(src)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        #x = self.backbone.avgpool(x)
        
        x = self.backbone.fc(x)
        x = x.permute(0, 3, 1, 2).flatten(2).permute(1, 0, 2)
        src_pad_mask = self.make_len_mask(x[:,:,0])
        src = self.pos_encoder(x)

        trg_pad_mask = self.make_len_mask(trg)
        trg = self.decoder(trg)
        trg = self.pos_decoder(trg)

        output = self.transformer(src, trg, src_mask=self.src_mask, tgt_mask=self.trg_mask, memory_mask=self.memory_mask,
                                  src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=trg_pad_mask, memory_key_padding_mask=src_pad_mask)
        output = self.fc_out(output)

        return output


# Обучение
def train(model, optimizer, criterion, iterator):
    model.train()
    epoch_loss = 0

    for (src, trg) in tqdm(iterator):
        src, trg = src.cuda(), trg.cuda()

        optimizer.zero_grad()
        output = model(src, trg[:-1,:])
        
        loss = criterion(output.view(-1, output.shape[-1]), torch.reshape(trg[1:,:], (-1,)))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        # Сохраняем картинки
        if random.random()<0.01:
            img = np.moveaxis(src[0].cpu().numpy(), 0, 2)
            imgplot = plt.imshow(img)
            plt.savefig('log/img/'+ labels_to_text(trg[1:,0].cpu().numpy(),idx2p))
    
    return epoch_loss / len(iterator)


def evaluate(model, criterion, iterator):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():    
        for (src, trg) in tqdm(iterator):
            src, trg = src.cuda(), trg.cuda()
            output = model(src, trg[:-1,:])
            loss = criterion(output.view(-1, output.shape[-1]), torch.reshape(trg[1:,:], (-1,)))
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def validate(model, dataloader, show=50):
    model.eval()
    show_count = 0
    error_w = 0
    error_p = 0
    with torch.no_grad():
        for (src, trg) in tqdm(dataloader):
            img = np.moveaxis(src[0].numpy(), 0, 2)
            src = src.cuda()
            x = model.backbone.conv1(src)
            x = model.backbone.bn1(x)
            x = model.backbone.relu(x)
            x = model.backbone.maxpool(x)
            x = model.backbone.layer1(x)
            x = model.backbone.layer2(x)
            x = model.backbone.layer3(x)
            x = model.backbone.layer4(x)
            #x = model.backbone.avgpool(x)
        
            x = model.backbone.fc(x)

            x = x.permute(0,3, 1, 2).flatten(2).permute(1, 0, 2)

            memory = model.transformer.encoder(model.pos_encoder(x))

            out_indexes = [p2idx['SOS'], ]

            for i in range(100):
                trg_tensor = torch.LongTensor(out_indexes).unsqueeze(1).to(device)

                output = model.fc_out(model.transformer.decoder(model.pos_decoder(model.decoder(trg_tensor)), memory))
                out_token = output.argmax(2)[-1].item()
                out_indexes.append(out_token)
                if out_token == p2idx['EOS']:
                    break

            out_p = labels_to_text(out_indexes[1:],idx2p)
            real_p = labels_to_text(trg[1:,0].numpy(),idx2p)
            error_w += int(real_p != out_p)
            if out_p:
                cer = phoneme_error_rate(real_p, out_p)
            else:
                cer = 1   
            
            error_p += cer     
            if show > show_count:
                #plt.imshow(img)
                #plt.show()
                show_count += 1
                print('Real:', real_p)
                print('Pred:', out_p)
                print(cer)
    
    return error_p/len(dataloader)*100, error_w/len(dataloader)*100

# Предсказания
def prediction():
    os.makedirs('/output', exist_ok=True)
    model.eval()
    
    with torch.no_grad():
        for filename in os.listdir(hp.test_dir):
            img = cv2.imread(hp.test_dir + filename,cv2.IMREAD_GRAYSCALE)#
            img = process_image(img).astype('uint8')
            img = img/img.max() 
            img = np.transpose(img,(2,0,1))

            src =  torch.FloatTensor(img).unsqueeze(0).cuda()

            x = model.backbone.conv1(src)
            x = model.backbone.bn1(x)
            x = model.backbone.relu(x)
            x = model.backbone.maxpool(x)

            x = model.backbone.layer1(x)
            x = model.backbone.layer2(x)
            x = model.backbone.layer3(x)
            x = model.backbone.layer4(x)
            #x = model.backbone.avgpool(x)
        
            x = model.backbone.fc(x)
            x = x.permute(0,3, 1, 2).flatten(2).permute(1, 0, 2)
            memory = model.transformer.encoder(model.pos_encoder(x))
            
            p_values = 1
            out_indexes = [p2idx['SOS'], ]
            for i in range(100):
                trg_tensor = torch.LongTensor(out_indexes).unsqueeze(1).to(device)
                output = model.fc_out(model.transformer.decoder(model.pos_decoder(model.decoder(trg_tensor)), memory))
                
                out_token = output.argmax(2)[-1].item()
                p_values = p_values * torch.sigmoid(output[-1, 0, out_token]).item()
                out_indexes.append(out_token)
                if out_token == p2idx['EOS']:
                    break
            
            pred = labels_to_text(out_indexes[1:],idx2p)
            print('pred:',p_values,pred)

            with open(os.path.join('/output', filename.replace('.jpg', '.txt').replace('.png', '.txt')), 'w', encoding="utf-8") as file:
                file.write(pred)

# Общая функция обучения и валидации
def train_all(best_eval_loss_cer):
    train_loss = 0
    count_bad = 0
    for epoch in range(epochs, 1000):
        print(f'Epoch: {epoch+1:02}')
        start_time = time.time()
        print("-----------train------------")
        train_loss = train(model, optimizer, criterion, train_loader)
        print("-----------valid------------")
        valid_loss = evaluate(model, criterion, val_loader)
        print("-----------eval------------")
        eval_loss_cer,eval_accuracy  = validate(model, val_loader, show=10)
        scheduler.step(eval_loss_cer)
        valid_loss_all.append(valid_loss)
        train_loss_all.append(train_loss)
        eval_loss_cer_all.append(eval_loss_cer)
        eval_accuracy_all.append(eval_accuracy)
        if eval_loss_cer < best_eval_loss_cer:
            count_bad = 0
            best_eval_loss_cer = eval_loss_cer
            torch.save({
                      'model': model.state_dict(),
                      'epoch': epoch,
                      'best_eval_loss_cer':best_eval_loss_cer,
                      'valid_loss_all': valid_loss_all,
                      'train_loss_all': train_loss_all,
                      'eval_loss_cer_all': eval_loss_cer_all,
                      'eval_accuracy_all': eval_accuracy_all,
                  }, './log/resnet50_trans_%.3f.pt'% (best_eval_loss_cer))
            print('Save best model')
        else:
            count_bad += 1
            torch.save({
                      'model': model.state_dict(),
                      'epoch': epoch,
                      'best_eval_loss_cer':best_eval_loss_cer,
                      'valid_loss_all': valid_loss_all,
                      'train_loss_all': train_loss_all,
                      'eval_loss_cer_all': eval_loss_cer_all,
                      'eval_accuracy_all': eval_accuracy_all,
                  }, './log/resnet50_trans_last.pt')
            print('Save model')
        print(f'Time: {time.time() - start_time}s')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val   Loss: {valid_loss:.4f}')
        print(f'Eval  CER: {eval_loss_cer:.4f}')
        print(f'Eval accuracy: {eval_accuracy:.4f}')
        plt.clf()
        plt.plot(valid_loss_all[-20:])
        plt.plot(train_loss_all[-20:])
        plt.savefig('log/all_loss.png')
        plt.clf()
        plt.plot(eval_loss_cer_all[-20:])
        plt.savefig('log/loss_cer.png')
        plt.clf()
        plt.plot(eval_accuracy_all[-20:])
        plt.savefig('log/eval_accuracy.png')
        if count_bad>19:
            break


# Загружаем гиперпараметры
hp = Hparams()

if __name__ == '__main__':
    # Фиксируем сиды
    random.seed(1488)
    torch.manual_seed(1488)
    torch.cuda.manual_seed(1488)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Создать папку с логами
    os.makedirs("log/img/", exist_ok=True)
    
    # Обучение или предсказание
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run", default='generate', help=\
        "Enter the function you want to run | Введите функцию, которую надо запустить (train, generate)")
    parser.add_argument("-c", "--checkpoint", default='', help="Чекпоинт")
    parser.add_argument("-d", "--test_dir", default='', help="Чекпоинт")
    
    args = parser.parse_args()
    if args.run == 'train' or args.run == 't':
        it_train = True
    else:
        it_train = False
    if args.checkpoint:
        hp.chk = args.checkpoint
    if args.test_dir:
        hp.test_dir = args.test_dir
    
    
    # Загрузить частоту слов
    if it_train:
        if hp.chk:
            hp.chk = './log/' + hp.chk
        
        # Загружаем название файлов, список строк для обучения и алфавит
        names,lines,cnt,all_word = process_texts(hp.image_dir,hp.trans_dir)  
        letters = set(cnt.keys())
        letters = sorted(list(letters))
        letters = ['PAD', 'SOS'] + letters + ['EOS']
    else:
        hp.chk = './' + hp.chk
        
        # Алфавит
        letters = ['PAD', 'SOS', ' ', '+', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9','[', ']', 
                'i', 'а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л','м', 'н', 'о', 'п', 'р', 
                'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я', 'ѣ', 'EOS']
   
    print('Символов:',len(letters),':', ' '.join(letters))

    # Перевод символов в индексы и наоборот
    p2idx = {p: idx for idx, p in enumerate(letters)}
    idx2p = {idx: p for idx, p in enumerate(letters)} 
    
    # Создадим обучающую и валидационную выборки.
    if it_train:
        
        lines_train = []
        names_train = []

        lines_val = []
        names_val = []

        for num,(line, name) in enumerate(zip(lines,names)):
            # файлы оканчивающиеся на 9 в валидацию
            if name[-5] == '9':
                lines_val.append(line)
                names_val.append(name)
            else:
                lines_train.append(line)
                names_train.append(name)

        image_train = generate_data(names_train,hp.image_dir) 
        image_val = generate_data(names_val,hp.image_dir) 
        
        # Датасеты
        train_dataset = TextLoader(image_train,lines_train,hp.image_dir, eval=False)
        train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True,
                                  batch_size=hp.batch_size, pin_memory=True,
                                  drop_last=True, collate_fn=TextCollate())

        val_dataset = TextLoader(image_val,lines_val,hp.image_dir, eval=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False,
                                                 batch_size=1, pin_memory=False,
                                                 drop_last=False, collate_fn=TextCollate())

    valid_loss_all, train_loss_all,eval_accuracy_all,eval_loss_cer_all = [],[],[],[]
    epochs, best_eval_loss_cer = 0, float('inf')
    
    # Создаём модель
    model = TransformerModel('resnet50', len(letters), hidden=hp.hidden, enc_layers=hp.enc_layers, dec_layers=hp.dec_layers, nhead = hp.nhead, dropout=hp.dropout, pretrained=it_train).to(device)
    
    # Загружаем веса
    if hp.chk:
        ckpt = torch.load(hp.chk)
        if 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
        else:
            model.load_state_dict(ckpt)
        if 'epochs' in ckpt:
            epochs = int(ckpt['epoch'])
        if 'valid_loss_all' in ckpt:
            valid_loss_all   = ckpt['valid_loss_all']
        if 'best_eval_loss_cer' in ckpt:
            best_eval_loss_cer   = ckpt['best_eval_loss_cer']        
        if 'train_loss_all' in ckpt:
            train_loss_all   = ckpt['train_loss_all']
        if 'eval_accuracy_all' in ckpt:
            eval_accuracy_all   = ckpt['eval_accuracy_all']
        if 'eval_loss_cer_all' in ckpt:
            eval_loss_cer_all   = ckpt['eval_loss_cer_all']
    
    optimizer = optim.AdamW(model.parameters(), lr=hp.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=p2idx['PAD'])
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 50, eta_min=0, last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    print(f'The model has {count_parameters(model):,} trainable parameters')
    #print(model)
    
    if it_train:
        train_all(best_eval_loss_cer)
    else:
        prediction()
