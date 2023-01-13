import collections
import os
import random
import time
from tqdm import tqdm
import torch
from torch import nn
import torchtext.vocab as Vocab
import torch.utils.data as Data


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def read_imdb(folder='train', data_root="data_root"):
    data = []
    for label in ['pos', 'neg']:
        folder_name = os.path.join(data_root, folder, label)
        for file in tqdm(os.listdir(folder_name)):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '').lower()
                data.append([review, 1 if label == 'pos' else 0])
    random.shuffle(data)
    return data

DATA_ROOT = os.getcwd()
data_root = os.path.join(DATA_ROOT, "aclImdb_v1", "aclImdb")
train_data, test_data = read_imdb('train', data_root), read_imdb('test', data_root)

# 打印训练数据中的前五个sample
for sample in train_data[:5]:
    print(sample[1], '\t', sample[0][:50])

def get_tokenized_imdb(data):  # 将每行数据的进行空格切割,保留每个的单词
    def tokenizer(text):
        return [tok.lower() for tok in text.split(' ')]
    return [tokenizer(review) for review, _ in data]

def get_vocab_imdb(data):

    tokenized_data = get_tokenized_imdb(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    # 统计所有的数据
    return Vocab.vocab(counter, min_freq=5)  # 构建词汇表,这里最小出现次数是5

vocab = get_vocab_imdb(train_data)

print('# words in vocab:', len(vocab))
# print(vocab[:5])
def preprocess_imdb(data, vocab):

    max_l = 500  # 将每条评论通过截断或者补0，使得长度变成500
    def pad(x):
        return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))
    tokenized_data = get_tokenized_imdb(data)
    features = torch.tensor([pad([vocab.get_stoi().get(word,0) for word in words]) for words in tokenized_data])
    # 填充,这里是将每一行数据扩充500个特征的
    labels = torch.tensor([score for _, score in data])
    return features, labels

train_set = Data.TensorDataset(*preprocess_imdb(train_data, vocab))
test_set = Data.TensorDataset(*preprocess_imdb(test_data, vocab))
batch_size = 64
train_iter = Data.DataLoader(train_set, batch_size, shuffle=True)
test_iter = Data.DataLoader(test_set, batch_size)

for X, y in train_iter:
    print('X', X.shape, 'y', y.shape)
    break
print('#batches:', len(train_iter))#391个批次,每个批次64个样本


class BiRNN(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers):
        super(BiRNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.encoder = nn.LSTM(input_size=embed_size,
                               hidden_size=num_hiddens,
                               num_layers=num_layers,
                               bidirectional=True)  # 双向循环网络
        self.decoder = nn.Linear(4 * num_hiddens, 2)

    def forward(self, inputs):
        embeddings = self.embedding(inputs.permute(1, 0))
        outputs, _ = self.encoder(embeddings)  # (seq_len, batch_size, 2*h)每一个输出,然后将第一次输出和最后一次输出拼接
        encoding = torch.cat((outputs[0], outputs[-1]), -1)  # (batch_size, 4*h)
        outs = self.decoder(encoding)  # (batch_size, 2)
        return outs

embed_size, num_hiddens, num_layers = 100, 100, 2
net = BiRNN(vocab, embed_size, num_hiddens, num_layers)

cache_dir = r"D:\Desktop\mycode\nlp_processing\glove_vocab"
glove_vocab = Vocab.GloVe(name='6B', dim=100, cache=cache_dir)

def load_pretrained_embedding(words, pretrained_vocab):
    embed = torch.zeros(len(words), pretrained_vocab.vectors[0].shape[0]) # 初始化为len*100维度
    oov_count = 0 # out of vocabulary
    for i, word in enumerate(words):
        try:
            idx = pretrained_vocab.stoi[word]
            embed[i, :] = pretrained_vocab.vectors[idx]# 将每个词语用训练的语言模型理解
        except KeyError:
            oov_count += 1
    if oov_count > 0:
        print("There are %d oov words." % oov_count)
    return embed
itos = vocab.get_itos()
net.embedding.weight.data.copy_(load_pretrained_embedding(itos, glove_vocab))
net.embedding.weight.requires_grad = False # 直接加载预训练好的, 所以不需要更新它

def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()
            else:
                if('is_training' in net.__code__.co_varnames):
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n

def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y) # 交叉熵损失函数
            optimizer.zero_grad()
            l.backward()
            optimizer.step()# 优化方法
            train_l_sum += l.cpu().item()# 进入cpu中统计
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

lr, num_epochs = 0.01, 5
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
loss = nn.CrossEntropyLoss()

train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)

def predict_sentiment(net, vocab, sentence):

    device = list(net.parameters())[0].device
    sentence = torch.tensor([vocab.get_stoi().get(word, 0) for word in sentence], device=device)
    label = torch.argmax(net(sentence.view((1, -1))), dim=1)
    return 'positive' if label.item() == 1 else 'negative'
result = predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'great'])
print(result)
