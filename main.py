import os,sys,torch,argparse,json
from torch import nn
from torchvision import datasets,transforms
from clipwrapper import ClipWrapper,ClipTransform
from custom_dataset import CustomDataset
from similarity_calculator import SimilarityCalculator
from tqdm import tqdm

class Integrator(nn.Module):
    def __init__(self,lvp_i,lvp_t):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.a = nn.parameter.Parameter(torch.ones(lvp_i.shape[0],1)*0.5)
        self.b = nn.parameter.Parameter(torch.ones(lvp_t.shape[0],1))
        self.lvp_i = lvp_i.to(self.device)
        self.lvp_t = lvp_t.to(self.device)

    def forward(self):
        lvp_it = self.a * self.lvp_t + self.b * self.lvp_i
        return lvp_it

def get_classes_from_file(path):
    current_path =  os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".")
    with open(current_path + path, 'r') as file: classes_original = json.loads(file.read())
    classes = list(classes_original.values())
    return classes

def cifar100_generate_image_embedings(args):
    cw = ClipWrapper(model_sel=args.model_sel)
    transform = ClipTransform(model_sel=args.model_sel)
    cifar100_train_dataset = datasets.CIFAR100(root='/datasets/CIFAR100',train=True,download=True,transform=transform)
    cifar100_test_dataset = datasets.CIFAR100(root='/datasets/CIFAR100',train=False,download=True,transform=transform)
    cifar100_train_loader = torch.utils.data.DataLoader(cifar100_train_dataset,batch_size=128,shuffle=False,num_workers=4)
    cifar100_test_loader = torch.utils.data.DataLoader(cifar100_test_dataset,batch_size=128,shuffle=False,num_workers=4)
    train_data = cw.inference_dataset(cifar100_train_loader)
    test_data = cw.inference_dataset(cifar100_test_loader)
    data = {'train':train_data,'test':test_data}
    torch.save(data,f'./data/cifar100_img_embeddings_{args.model_sel}.pt')

def cifar100_generate_lvp_i(args):
    data = torch.load(f'./data/cifar100_img_embeddings_{args.model_sel}.pt')
    train_data = data['train']
    lvp_list = []
    
    for i in range(100):
        data_i = train_data['data'][train_data['targets'] == i]
        lvp_i = torch.mean(data_i,dim=0)
        lvp_list.append(lvp_i)
    lvp = torch.stack(lvp_list,dim=0)
    print(lvp.shape)
    torch.save(lvp,f'./data/cifar100_lvp_i_{args.model_sel}.pt')

def cifar100_generate_lvp_t(args):
    classes = get_classes_from_file('/data/cifar100_classes.txt')
    cw = ClipWrapper(model_sel=args.model_sel,classes=classes,base_text='a photo of a',generate_text=True)
    text_embeddings = cw.text_features.to('cpu')
    print(text_embeddings.shape)
    torch.save(text_embeddings,f'./data/cifar100_lvp_t_{args.model_sel}.pt')

def cifar100_generate_lvp_it(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lvp_i = torch.load(f'./data/cifar100_lvp_i_{args.model_sel}.pt').to(device)
    lvp_t = torch.load(f'./data/cifar100_lvp_t_{args.model_sel}.pt').to(device)
    sc = SimilarityCalculator()
    
    model = Integrator(lvp_i,lvp_t).to(device)
    optimizer = torch.optim.SGD(model.parameters(),lr=0.0001)
    loss_fn = nn.CrossEntropyLoss()
    
    data = torch.load(f'./data/cifar100_img_embeddings_{args.model_sel}.pt')
    train_data = data['train']
    train_dataset_loader = torch.utils.data.DataLoader(CustomDataset(train_data['data'],train_data['targets']),batch_size=256,shuffle=False,num_workers=4)
    
    for epoch in range(100):
        avg_loss = 0
        for image_features, labels in tqdm(train_dataset_loader):
            image_features = image_features.to(device)
            labels = labels.to(device)
            lvp_it= model()
            values, indices, similarity = sc(lvp_it,image_features,dis_func='L1')
            loss = loss_fn(similarity,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
        print(f'epoch: {epoch}, loss: {avg_loss / len(train_dataset_loader)}')
    lvp_it = model()
    torch.save(lvp_it.detach().cpu(),f'./data/cifar100_lvp_it_{args.model_sel}.pt')

def cifar100_generate_lvp_c(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lvp_it = torch.load(f'./data/cifar100_lvp_it_{args.model_sel}.pt')
    labels = torch.arange(100)
    
    model = nn.Linear(lvp_it.shape[1],100).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    
    dataset_loader = torch.utils.data.DataLoader(CustomDataset(lvp_it,labels),batch_size=100,shuffle=False)
    
    for epoch in range(150):
        avg_loss = 0
        for lvp_it_batch, labels_batch in tqdm(dataset_loader):
            lvp_it_batch = lvp_it_batch.to(device)
            labels_batch = labels_batch.to(device)
            outputs = model(lvp_it_batch)
            loss = loss_fn(outputs,labels_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
        print(f'epoch: {epoch}, loss: {avg_loss / len(dataset_loader)}')
    torch.save(model.state_dict(),f'./data/cifar100_lvp_c_{args.model_sel}.pt')



def cifar100_eval_lvp(args):
    cw = ClipWrapper(model_sel=args.model_sel)
    data = torch.load(f'./data/cifar100_img_embeddings_{args.model_sel}.pt')
    lvp = torch.load(f'./data/cifar100_lvp_{args.lvp}_{args.model_sel}.pt')
    test_data = data['test']
    test_dataset_loader = torch.utils.data.DataLoader(CustomDataset(test_data['data'],test_data['targets']),batch_size=256,shuffle=False,num_workers=4)
    acc,num_data = cw.eval_dataset_lvp(lvp,torch.arange(100),test_dataset_loader)

def cifar100_eval_lvp_c(args):
    data = torch.load(f'./data/cifar100_img_embeddings_{args.model_sel}.pt')
    lvp_it = torch.load(f'./data/cifar100_lvp_it_{args.model_sel}.pt')
    model = nn.Linear(lvp_it.shape[1],100)
    model.load_state_dict(torch.load(f'./data/cifar100_lvp_c_{args.model_sel}.pt'))
    model.eval()
    test_data = data['test']
    test_dataset_loader = torch.utils.data.DataLoader(CustomDataset(test_data['data'],test_data['targets']),batch_size=256,shuffle=False,num_workers=4)
    acc,num_data = 0,0
    for image_features, labels in tqdm(test_dataset_loader):
        image_features = image_features.to('cpu')
        labels = labels.to('cpu')
        outputs = model(image_features)
        predicted = torch.argmax(outputs, dim=1)
        acc += (predicted == labels).sum().item()
        num_data += len(labels)
    acc = acc / num_data
    print(f'accuracy: {acc}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--function', type=str)
    parser.add_argument('-m','--model_sel', type=int, default=5)
    parser.add_argument('-lvp', type=str, default='i')
    args = parser.parse_args()
    getattr(sys.modules[__name__], args.function)(args)