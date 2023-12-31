import os,sys
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn
import torch
import copy
import json
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
# todo 每次训练要改slide model保存路径
class Config:
    folder = '/root/data'    # 数据集所在文件夹
    slide = 32
    batch_size = 32
    savename = 'bert_lstm'    # 名称
    path = f'/root/12{savename}_model.pth'        # 模型加载路径
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    learning_rate = 1e-5             # 学习率
    epochs = 5
    pred_len = 200
    is_lock = False          # 选择训练训练bert的哪些层
    is_test = True          # 确定是否进行小说生成预测  测试

# 预处理  输出是 32连续字符 label是后面接着的字  每句话隔16的字符，再次选择
def make_dataset(folder, slide):
    dirs = os.listdir(folder)
    x = []
    y = []
    for sub_folder in dirs:
        for path in os.listdir(folder + '/' + sub_folder):
            path1 = folder + '/' + sub_folder + '/' + path
            data = read_txt(path1)
            #print(data)
            if len(data) < slide + 1:
                continue
            lenth = int(config.batch_size / 2)
            for i in range(len(data) // lenth - 32):
                x.append(data[i * lenth:i* lenth + slide])
                y.append(data[i * lenth + slide])
                print(data[i * lenth:i* lenth + slide], data[i * lenth + slide])
            print(sub_folder + ' ' + path.replace('.txt', ' ') + '已经读取完毕')
    return x,y

# 读txt文件 输出无空格和换行的纯文本
def read_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.readlines()
        if len(content) >= 2 :
            content = content[1].replace(u'\xa0', u'').replace(u'\u3000\u3000', u'').replace('\n','').replace(' ','')
        else:
            content = ''
            return content
    return content

from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
bert = BertModel.from_pretrained("bert-base-chinese")

class Mydataset(Dataset):
    def __init__(self,data,label):
        self.x = data
        self.y = label
    def __getitem__(self, index):
        inputs = tokenizer(self.x[index], max_length=config.slide+2, truncation=True ,padding='max_length')
        input_ids = torch.tensor(inputs['input_ids'])
        attention_masks = torch.tensor(inputs['attention_mask'])
        token_type_ids = torch.tensor(inputs['token_type_ids'])
        target = torch.tensor(int(tokenizer.convert_tokens_to_ids(self.y[index])))
        return input_ids,attention_masks,token_type_ids,target
    def __len__(self):
        return  len(self.x)

# LSTM 模型
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1 # 单向LSTM
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        # 分类的全链接层
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        #  batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        #  前向传播过程新生成的变量，需要传递到device中去
        # 如果是测试  batch_size = 1
        if input_seq.size(0) == 1:
            self.batch_size = 1
        h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(config.device)
        c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(config.device)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(input_seq, (h_0, c_0)) # output(bs, seq_len, hidden_size)
        pred = self.linear(output)  # (bs, seq_len, output_size)
        pred = pred[:, -1, :]  # (bs, output_size)
        return pred
class Bert_Fc_Model(nn.Module):
    def __init__(self,is_lock=False):
        super(Bert_Fc_Model,self).__init__()
        self.bert_pretrained = bert
        self.fc = nn.Linear(768,512)
        self.fc1 = nn.Linear(512, len(tokenizer))
        self.dropout = nn.Dropout()
        # self.lstm = LSTM(input_size=512, hidden_size=256, num_layers=1, output_size=len(tokenizer), batch_size=batch_size)
        if is_lock:
            # 加载并冻结bert模型参数
            for name, param in self.bert_pretrained.named_parameters():
                if name.startswith('pooler'):
                    continue
                else:
                    param.requires_grad_(False)

    def forward(self,x,attention_masks=None,token_type_ids=None):
        x = self.bert_pretrained(x,attention_masks,token_type_ids)['last_hidden_state']
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        x = self.fc1(x)
        pred = x[:, -1, :]
        return pred
class Bert_Lstm_Model(nn.Module):
    def __init__(self,is_lock=False):
        super(Bert_Lstm_Model,self).__init__()
        self.bert_pretrained = bert
        self.fc = nn.Linear(768, 512)
        self.fc1 = nn.Linear(512, len(tokenizer))
        self.dropout = nn.Dropout()
        self.lstm = LSTM(input_size=512, hidden_size=512, num_layers=3, output_size=len(tokenizer),
                         batch_size=config.batch_size)
        if is_lock:
            # 加载并冻结bert模型参数
            for name, param in self.bert_pretrained.named_parameters():
                if name.startswith('pooler'):
                    continue
                else:
                    param.requires_grad_(False)

    def forward(self, x, attention_masks=None, token_type_ids=None):
        x = self.bert_pretrained(x, attention_masks, token_type_ids)['last_hidden_state']
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        x = self.lstm(x)
        return x

# 训练文件
def train(epochs):
    for epoch in range(1, epochs + 1):
        model.train()
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        running_loss = 0.0
        total = 0
        right = 0
        for batch_idx, (input_ids, attention_masks, token_type_ids, target) in loop:
            total += 1
            input_ids, attention_masks, token_type_ids, target = input_ids.to(config.device), attention_masks.to(
                config.device), token_type_ids.to(config.device), target.to(config.device)
            optimizer.zero_grad()
            output = model(input_ids, attention_masks, token_type_ids)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            right += accuracy(output, target)

            loop.set_description(f'Epoch [{epoch}/{epochs}]')
            loop.set_postfix(loss=running_loss / (batch_idx + 1),
                             acc=float(right) / float(config.batch_size * batch_idx + len(input_ids)))

        train_losses.append(running_loss / total)
        # 开始测试
#         model.eval()
#         test_loss = 0
#         correct = 0
#         with torch.no_grad():
#             for input_ids, attention_masks, token_type_ids, target in test_loader:
#                 input_ids, attention_masks, token_type_ids, target = input_ids.to(device), attention_masks.to(
#                     device), token_type_ids.to(device), target.to(device)
#                 output = model(input_ids, attention_masks, token_type_ids)
#                 test_loss += F.cross_entropy(output, target, size_average=False).item()
#                 correct += accuracy(output, target)

#         test_loss /= len(test_loader.dataset)
#         test_losses.append(test_loss)
#         print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
#             test_loss, correct, len(test_loader.dataset),

#             100. * correct / len(test_loader.dataset)))
        # 测试结束
        # if correct / len(test_loader.dataset) > max_acc:
        #     max_acc = correct / len(test_loader.dataset)
        torch.save(model.state_dict(), config.path)
        best_model_wts = copy.deepcopy(model.state_dict())
# 准确率计算函数
def accuracy(predictions,labels):
    pred = torch.max(predictions.data,1,keepdim=True)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights
# 文本生成预测 函数
def test(sentence,pred_len):
    present = []
    print('输入句子: ',sentence)
    for i in range(pred_len):
        inputs = tokenizer(sentence[-32:], max_length=config.slide + 2, truncation=True, padding='max_length')
        input_ids = torch.tensor(inputs['input_ids']).unsqueeze(0)
        pred = model(input_ids.to(config.device))
        # print('topk顺序如下')
        k = 10
        topk = torch.topk(torch.softmax(pred,dim=-1),k=k,dim=-1,largest=True)[1].cpu().numpy()
        # result 预测出来的编号
        result = 0
        # 找到不重复的
        if len(present) == 30:  # 30个字内不能重复
            present.remove(present[0])
        for i in topk[0,:]:
            if i == 100 or i == 0:    # 跳过特殊字符
                continue
            elif present.count(i) > 0:   # 重复惩罚: 30个词内不能重复
                continue
            else:              #todo 束搜集
                result = i
                present.append(i)
                break
            # 非[UNK]
        sentence = sentence + tokenizer.decode(result)
    print('续写完成后的内容是：', sentence)
# 配置类Config

if __name__ == "__main__":
    #todo 设置相关变量和参数
    config = Config
    # 选择进行  测试or训练
    if config.is_test:
        print("模型权重路径是：", config.path)
        print("正在载入模型准备测试......................................................")
        sentence = '围绕这座山峰的是一圈又一圈建筑，它们或倒塌于地，或被烧得漆黑，让人看不出完好时是什'
        # todo 选择模型名称   准备开始测试
        model = Bert_Lstm_Model()
        model.load_state_dict(torch.load(config.path))
        model.to(config.device)
        test(sentence, config.pred_len)
    else:
        # 定义数据集
        data, label = make_dataset(config.folder, config.slide)
        print('输出处理完成 一共{%d}数据 ' % len(data))
        train_dataset = Mydataset(data, label)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
        print('*' * 30)
        print(' 数据集全部处理完毕 ')
        #todo 选择模型名称   设置模型相关信息
        model = Bert_Lstm_Model()
        criterion = F.cross_entropy
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        model.to(config.device)
        train_losses = []
        test_losses = []
        max_acc = 0
        print(config.path)
        # 不存在权重参数， 从头训练新的模型
        if not os.path.exists(config.path):
            print('不存在路径，从头开始训练模型')
            train(config.epochs)
        # 加载已有模型的， 接着训练
        else:
            print('路径已有,加载模型ing',config.path)
            model.load_state_dict(torch.load(config.path))
            model.to(config.device)
            train(config.epochs)
