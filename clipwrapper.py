import os,sys,torch,pickle,pathlib
from tqdm import tqdm
from similarity_calculator import SimilarityCalculator
from local.clip import* 

class ClipTransform:
    def __init__(self,model_sel) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.available_models = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64','ViT-B/16','ViT-B/32', 'ViT-L/14', 'ViT-L/14@336px']
        self.model_name = self.available_models[model_sel]
        _, self.preprocess = clip.load(self.model_name, self.device)

    def __call__(self, pic):
        pic = self.preprocess(pic)
        return pic

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class ClipWrapper():
    def __init__(self,model_sel=5,classes=[''],base_text='',generate_text=True) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_sel = model_sel
        self.available_models = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/16', 'ViT-B/32', 'ViT-L/14', 'ViT-L/14@336px']
        #print(clip.available_models())
        self.parameters = ['38,316,896','56,259,936','87,137,080','167,328,912','420,380,352','86,192,640','87,849,216','303,966,208','304,293,888']
        self.model_name = self.available_models[model_sel]
        print("Clip: " + self.model_name)
        self.model, self.preprocess = clip.load(self.model_name, self.device)
        total_params = sum(p.numel() for p in self.model.visual.parameters())
        print(f'total parameters: {total_params:,}')
        self.classes = classes
        self.base_text = base_text
        print('base text: ' + self.base_text)
        self.similarity_calculator = SimilarityCalculator()
        if generate_text: self.generate_text_features()
    
    def generate_text_features(self):
        print('classes:')
        print(self.classes)
        text_inputs = torch.cat([clip.tokenize(self.base_text + ' ' + c) for c in self.classes]).to(self.device)
        with torch.no_grad(): self.text_features = self.model.encode_text(text_inputs)
        self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
    
    def generate_img_features(self,images):
        image_input = images.to(self.device) if len(images.shape) == 4 else images.unsqueeze(0).to(self.device)
        with torch.no_grad(): image_features = self.model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features
    
    def get_predictions(self,images,dis_func):
        image_features = self.generate_img_features(images)
        values, indices,similarity = self.similarity_calculator(self.text_features,image_features,dis_func=dis_func)
        return values, indices
    
    def eval_dataset_text(self,dataset,dis_func='Cos'):
        acc,nums = 0,0
        for images, labels in tqdm(dataset):
            values, indices = self.get_predictions(images,dis_func=dis_func)
            indices = torch.flatten(indices)
            result = torch.eq(labels.to(self.device),indices.to(self.device))
            acc += torch.sum(result)
            nums += len(labels)
        acc = acc.to(torch.float) / nums
        print(acc)
        return acc
    
    def eval_dataset_lvp(self,lvp,label,dataset_loader,dis_func='Cos'):
        acc,num_data = 0,0
        label = label.to(self.device)
        for image_features, labels in tqdm(dataset_loader):
            values, indices, similarity = self.similarity_calculator(lvp,image_features,dis_func=dis_func)
            indices = torch.flatten(indices)
            indices = label[indices]
            result = torch.eq(labels.to(self.device),indices.to(self.device))
            acc += torch.sum(result)
            num_data += len(labels)
        acc = acc.to(torch.float) / num_data
        print('accuracy: ',acc)
        return acc,num_data
    
    def inference_dataset(self,dataset_loader):
        data  = {'data':[],'targets':[]}
        for images, labels in tqdm(dataset_loader):
            img_features = self.generate_img_features(images)
            
            data['data'].append(img_features.cpu().type(torch.float))
            data['targets'].append(labels.cpu().type(torch.long))
        
        data['data'] = torch.cat(data['data'],0)
        data['targets'] = torch.cat(data['targets'],0)
        return data