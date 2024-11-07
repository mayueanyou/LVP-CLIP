import os,sys,torch
import torch.nn.functional as F

class SimilarityCalculator():
    def __init__(self,topk=1,use_cdist=True) -> None:
        self.topk = topk
        self.use_cdist = use_cdist
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def __call__(self,label_features,input_features,dis_func='L1'): #[n,d], [x,d]
        label_features = label_features.to(self.device)
        input_features = input_features.to(self.device)
        if dis_func=='L1': return self.l1(label_features,input_features)
        elif dis_func=='L2': return self.l2(label_features,input_features)
        elif dis_func=='Mul': return self.mul(label_features,input_features)
        elif dis_func=='Cos': return self.cos(label_features,input_features)
        else: return self.p_norm(label_features,input_features,dis_func)
    
    def topk_similarity(self,similarity):
        similarity = similarity.softmax(dim=-1)
        values, indices = similarity.topk(self.topk)
        return values, indices, similarity
    
    def p_norm(self,label_features,input_features,p):
        similarity = torch.cdist(input_features,label_features,p=p)
        similarity = torch.neg(similarity)
        return self.topk_similarity(similarity)
    
    def l1(self,label_features,input_features):
        if self.use_cdist: similarity = torch.cdist(input_features,label_features,p=1)
        else: similarity = self.L1_similarity(label_features,input_features)
        similarity = torch.neg(similarity)
        return self.topk_similarity(similarity)
        
    def l2(self,label_features,input_features):
        if self.use_cdist: similarity = torch.cdist(input_features,label_features,p=2)
        else: self.L2_similarity(label_features,input_features)
        similarity = torch.neg(similarity)
        return self.topk_similarity(similarity)
    
    def mul(self,label_features,input_features):
        similarity = input_features @ label_features.T
        return self.topk_similarity(similarity)
    
    def cos(self,label_features,input_features):
        similarity = self.Cosine_similarity(label_features, input_features)
        return self.topk_similarity(similarity)
    
    def Cosine_similarity(self,label_features,input_features):
        input_features = input_features.unsqueeze(1)
        input_features = input_features.repeat(1,label_features.size(0),1)
        similarity = F.cosine_similarity(input_features,label_features,dim=2)
        return similarity
    
    def L1_similarity(self,label_features,input_features):
        input_features = input_features.unsqueeze(1)
        input_features = input_features.repeat(1,label_features.size(0),1)
        similarity = torch.abs(input_features - label_features)
        similarity = torch.sum(similarity,-1)
        return similarity
    
    def L2_similarity(self,label_features,input_features):
        input_features = input_features.unsqueeze(1)
        input_features = input_features.repeat(1,label_features.size(0),1)
        similarity = torch.pow(input_features - label_features,2)
        similarity = torch.sum(similarity,-1)
        return similarity
