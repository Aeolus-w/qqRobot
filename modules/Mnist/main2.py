import torch
import torchvision 
from tqdm import tqdm
import matplotlib.pyplot 
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.model = torch.nn.Sequential(
            #The size of the picture is 28*28
            torch.nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, stride = 1, padding = 1),
            torch.nn.ReLU(),#Rectified Linear Unit 修正线性单元，=max（x，0）
            torch.nn.MaxPool2d(kernel_size = 2,stride = 2),#pool，汇聚，抓住主要矛盾
            
            #The size of the picture is 14*14
            torch.nn.Conv2d(in_channels = 16,out_channels = 32,kernel_size = 3,stride = 1,padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2,stride = 2),
            
            #The size of the picture is 7*7
            torch.nn.Conv2d(in_channels = 32,out_channels = 64,kernel_size = 3,stride = 1,padding = 1),
            torch.nn.ReLU(),

            # The size of the picture is 3x3
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # 新增池化层
            
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=3 * 3 * 128, out_features=128),  
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=10),
            torch.nn.Softmax(dim=1)
            #Softmax 函数用于将网络的输出变为概率，且所有类别的概率之和为 1。dim=1 表示 Softmax 操作是在每个样本的维度（每一行）上进行操作。输出的每个数值代表属于某一类的概率，概率最大的类别会被选为模型的预测结果
        )
    
    def forward(self, input):
        output = self.model(input)
        return output

device =  "cuda:0" if torch.cuda.is_available() else "cpu"
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean = [0.5],std = [0.5])])

BATCH_SIZE = 256
EPOCHS = 10
trainData = torchvision.datasets.MNIST('./data/',train = True,transform = transform,download = True)
testData = torchvision.datasets.MNIST('./data/',train = False,transform = transform)


trainDataLoader = torch.utils.data.DataLoader(dataset = trainData,batch_size = BATCH_SIZE,shuffle = True)
testDataLoader = torch.utils.data.DataLoader(dataset = testData,batch_size = BATCH_SIZE)
net = Net()
print(net.to(device))

lossF = torch.nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(net.parameters())

history = {'Test Loss':[],'Test Accuracy':[]}
for epoch in range(1, EPOCHS + 1):
    processBar = tqdm(trainDataLoader, unit = 'step')
    net.train(True)
    for step,(trainImgs, labels) in enumerate(processBar):
        trainImgs = trainImgs.to(device)
        labels = labels.to(device)

        net.zero_grad()
        outputs = net(trainImgs)
        loss = lossF(outputs, labels)
        predictions = torch.argmax(outputs, dim = 1)
        accuracy = torch.sum(predictions == labels) / labels.shape[0]
        loss.backward()

        optimizer.step()
        processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f" % (epoch,EPOCHS,loss.item(),accuracy.item()))

        if step == len(processBar) - 1:
            correct, totalLoss = 0,0
            net.train(False)
            with torch.no_grad():
                for testImgs, labels in testDataLoader:
                    testImgs = testImgs.to(device)
                    labels = labels.to(device)
                    outputs = net(testImgs)
                    loss = lossF(outputs, labels)
                    predictions = torch.argmax(outputs, dim = 1)

                    totalLoss +=loss
                    correct += torch.sum(predictions == labels)

                    testAccuracy = correct/(BATCH_SIZE * len(testDataLoader))
                    testLoss = totalLoss/len(testDataLoader)
                    history['Test Loss'].append(testLoss.item())
                    history['Test Accuracy'].append(testAccuracy.item())
            
            processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f" % (epoch,EPOCHS,loss.item(),accuracy.item(),testLoss.item(),testAccuracy.item()))
    processBar.close()

matplotlib.pyplot.plot(history['Test Loss'],label = 'Test Loss')
matplotlib.pyplot.legend(loc='best')
matplotlib.pyplot.grid(True)
matplotlib.pyplot.xlabel('Epoch')
matplotlib.pyplot.ylabel('Loss')
matplotlib.pyplot.show()

matplotlib.pyplot.plot(history['Test Accuracy'],color = 'red',label = 'Test Accuracy')
matplotlib.pyplot.legend(loc='best')
matplotlib.pyplot.grid(True)
matplotlib.pyplot.xlabel('Epoch')
matplotlib.pyplot.ylabel('Accuracy')
matplotlib.pyplot.show()

torch.save(net,'./result/model.pth')

model = Net()

def test_mydata():
    image = Image.open('./test/4.png')   #读取自定义手写图片
    image = image.resize((28, 28))   # 裁剪尺寸为28*28
    image = image.convert('L')  # 转换为灰度图像
    transform = transforms.ToTensor()
    image = transform(image)
    image = image.resize(1, 1, 28, 28)
    output = model(image)
    probability, predict = torch.max(output.data, dim=1)
    print("此手写图片值为：%d,其最大概率为：%.2f " % (predict[0], probability))
    plt.title("此手写图片值为：{}".format((int(predict))), fontname='SimHei')
    plt.imshow(image.squeeze())
    plt.show()
    

# 测试主函数
if __name__ == '__main__':
    test_mydata()