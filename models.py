"""
Deep Graph Infomax in DGL

References
----------
Papers: https://arxiv.org/abs/1809.10341
Author's code: https://github.com/PetarV-/DGI
"""
#np.argpartition
from tqdm import tqdm
import torch
import torch.nn as nn
import math
from gcn import GCN
import torch.nn.functional as F
from sampler import ClusterIter
from sklearn.metrics import f1_score
class LinkPredLoss(nn.Module):
    def __init__(self):
        super(LinkPredLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()
    def forward(self, edges, cluster_logits):
        # chosen = torch.randint(edges[0].shape[0],(1000,))
        src = cluster_logits[edges[0]]
        tar = cluster_logits[edges[1]]
        neg = cluster_logits[torch.randint(cluster_logits.shape[0],(src.shape[0],))]
        pos_score = (src*tar).sum(-1)
        neg_score = (src*neg).sum(-1)
        link_pred_loss = self.loss(pos_score,torch.ones_like(pos_score))+\
                            self.loss(neg_score,torch.zeros_like(neg_score))
        avg_loss = torch.log(cluster_logits.mean(0)+0.0001).mean()                   
        # avg_loss = torch.log(cluster_logits.max(0)[0]+0.0001).mean()
        loss = link_pred_loss - avg_loss
        return loss
class Encoder(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers, activation, dropout):
        super(Encoder, self).__init__()
        self.conv = GCN(in_feats, n_hidden, n_hidden, n_layers, activation, dropout)

    def forward(self, g, features, corrupt=False):
        if corrupt:
            perm = torch.randperm(g.number_of_nodes())
            features = features[perm]
        features = self.conv(g, features)
        return features


class Discriminator(nn.Module):
    def __init__(self, n_hidden):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.reset_parameters()

    def uniform(self, size, tensor):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
        size = self.weight.size(0)
        self.uniform(size, self.weight)

    def forward(self, features, targets):
        scores = torch.matmul(features, torch.matmul(self.weight, targets.t()))
        return scores# shape: n*n_targets

class Classifier(nn.Module):
    def __init__(self, n_hidden, n_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(n_hidden, n_classes)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc.reset_parameters()

    def forward(self, features):
        features = self.fc(features)
        # return torch.sigmoid(features)
        return torch.log_softmax(features, dim=-1)

class DGIclassifier(nn.Module):
    def __init__(self, data_info, args):
        super(DGIclassifier, self).__init__()
        self.encoder = Encoder(data_info.in_feats, args.n_hidden, args.n_layers, nn.PReLU(args.n_hidden), args.dropout)
        self.discriminator = Discriminator(args.n_hidden)
        self.classifier = Classifier(args.n_hidden, data_info.n_classes)
        self.loss_dgi = nn.BCEWithLogitsLoss()
        if args.gpu >= 0:
            self.cuda()
    def forward(self, g, classifier = False):
        features = g.ndata['feat']
        positive_h = self.encoder(g, features, corrupt=False)
        negative_h = self.encoder(g, features, corrupt=True)
        summary = torch.sigmoid(positive_h.mean(dim=0,keepdim=True))
        positive_score_global = self.discriminator(positive_h, summary)
        negative_score_global = self.discriminator(negative_h, summary)
        loss_dgi = \
            self.loss_dgi(positive_score_global, torch.ones_like(positive_score_global))+\
            self.loss_dgi(negative_score_global, torch.zeros_like(negative_score_global))
        #################################################################################
        # cluster_logits = self.cluster(positive_h.detach())
        # loss_cluster = self.loss_link(g.edges(), cluster_logits)
        #################################################################################
        loss_class = None
        if classifier:
            train_mask = g.ndata['train_mask']
            labels = g.ndata['label']
            preds = self.classifier(positive_h)
            loss_class = F.nll_loss(preds[train_mask], labels[train_mask])

        
        ##################################################################################
        return loss_dgi, loss_class
    def get_embedding(self, g, args):
        if args.gpu >= 0:
            g = g.to(args.gpu)
        with torch.no_grad():
            h = self.encoder(g, g.ndata['feat'], corrupt=False)
        return h.detach()
    def get_embedding_batch(self, g, args):
        embeddings = torch.empty((g.num_nodes(),args.n_hidden))
        g_iterator = ClusterIter(args.dataset, g, args.psize, args.batch_size, None)
        for g_sub in g_iterator:
            if args.gpu >= 0:
                g_sub = g_sub.to(args.gpu)
            with torch.no_grad():
                h = self.encoder(g_sub, g_sub.ndata['feat'], corrupt=False)
            embeddings[g_sub.ndata['_ID']] = h.detach().cpu()
        return embeddings
    def get_class(self, g, args):
        if args.gpu >= 0:
            g = g.to(args.gpu)
        with torch.no_grad():
            h = self.encoder(g, g.ndata['feat'], corrupt=False)
            class_logits = self.classifier(h)
        return class_logits.detach()
    def train_dgi(self, g, args):
        optimizer_dgi = torch.optim.Adam([{'params':self.encoder.parameters()},
                                          {'params':self.discriminator.parameters()}],
                                         lr=args.dgi_lr,
                                         weight_decay=args.weight_decay)
        cnt_wait = 0
        best = 1e9
        for epoch in range(args.n_dgi_epochs):
            if args.gpu >= 0:
                g = g.to(args.gpu)
            self.encoder.train()
            self.discriminator.train()
            optimizer_dgi.zero_grad()
            loss_dgi, _ = self.forward(g)
            loss_dgi.backward()
            optimizer_dgi.step()
            loss = loss_dgi.item()
            if loss < best:
                best = loss
                cnt_wait = 0
                torch.save(self.state_dict(), './model_save/'+args.dataset+'/best_dgi.pkl')
            else:
                cnt_wait += 1
        
            if cnt_wait == args.patience:
                # print('Early stopping!')
                break
            if args.report and (epoch%5==0):
                print("Epoch {:05d}|loss_dgi {:.4f}"\
                      .format(epoch, loss))
        self.load_state_dict(torch.load('./model_save/'+args.dataset+'/best_dgi.pkl'))
    def train_classifier(self, g, args):
        
        optimizer_class = torch.optim.Adam([{'params':self.encoder.parameters(),'lr':args.classifier_lr*0.1},
                                          {'params':self.classifier.parameters(),'lr':args.classifier_lr}],
                                         weight_decay=args.weight_decay)
        cnt_wait = 0
        best = 0
        for epoch in range(args.n_classifier_epochs):
            if args.gpu >= 0:
                g = g.to(args.gpu)
            self.encoder.train()
            self.discriminator.train()
            optimizer_class.zero_grad()
            loss_dgi, loss_class = self.forward(g, True)
            loss = loss_dgi + loss_class
            loss.backward()
            optimizer_class.step()
            
            acc_val,_ = self.evaluate(g, 'val_mask',args)
            if acc_val > best:
                best = acc_val
                cnt_wait = 0
                torch.save(self.state_dict(), './model_save/'+args.dataset+'/best_model.pkl')
            else:
                cnt_wait += 1
        
            if cnt_wait == args.patience:
                # print('Early stopping!')
                break
            if args.report and (epoch%5==0):
                print("Epoch {:05d}|loss_dgi {:.4f}|loss_class {:.4f}"\
                      .format(epoch, loss_dgi.item(),loss_class.item()))
        self.load_state_dict(torch.load('./model_save/'+args.dataset+'/best_model.pkl'))
    def train_dgi_batch(self, g, args):
        g_iterator = ClusterIter(args.dataset, g, args.psize, args.batch_size, None)
        optimizer_dgi = torch.optim.Adam([{'params':self.encoder.parameters()},
                                          {'params':self.discriminator.parameters()}],
                                         lr=args.dgi_lr,
                                         weight_decay=args.weight_decay)
        cnt_wait = 0
        best = 1e9
        for epoch in range(args.n_dgi_epochs):
            loss_list =  []
            for g_sub in g_iterator:
                if args.gpu >= 0:
                    g_sub = g_sub.to(args.gpu)
                self.encoder.train()
                self.discriminator.train()
                optimizer_dgi.zero_grad()
                loss_dgi, _ = self.forward(g_sub)
                loss_dgi.backward()
                optimizer_dgi.step()
                loss_list.append(loss_dgi.item())
            loss = torch.mean(torch.tensor(loss_list,dtype=float))
            if loss < best:
                best = loss
                cnt_wait = 0
                torch.save(self.state_dict(), './model_save/'+args.dataset+'/best_dgi.pkl')
            else:
                cnt_wait += 1
        
            if cnt_wait == args.patience:
                # print('Early stopping!')
                break
            if args.report and (epoch%5==0):
                print("Epoch {:05d}|loss_dgi {:.4f}"\
                      .format(epoch, loss))
        self.load_state_dict(torch.load('./model_save/'+args.dataset+'/best_dgi.pkl'))
    def train_classifier_batch(self, g, args):
        g_iterator = ClusterIter(args.dataset, g, args.psize, args.batch_size, None)
        optimizer_class = torch.optim.Adam([{'params':self.encoder.parameters(),'lr':args.classifier_lr*0.1},
                                          {'params':self.classifier.parameters(),'lr':args.classifier_lr}],
                                         weight_decay=args.weight_decay)
        cnt_wait = 0
        best = 0
        for epoch in range(args.n_classifier_epochs):
            acc_list =  []
            for g_sub in g_iterator:
                if g_sub.ndata['train_mask'].sum()==0:
                    continue
                if args.gpu >= 0:
                    g_sub = g_sub.to(args.gpu)
                self.encoder.train()
                self.discriminator.train()
                optimizer_class.zero_grad()
                loss_dgi, loss_class = self.forward(g_sub, True)
                loss = loss_dgi + loss_class
                loss.backward()
                optimizer_class.step()
                
                acc_list.append(self.evaluate(g_sub, 'val_mask',args)[0])
            acc_val = torch.mean(torch.tensor(acc_list,dtype=float)).item()
            if acc_val > best:
                best = acc_val
                cnt_wait = 0
                torch.save(self.state_dict(), './model_save/'+args.dataset+'/best_model.pkl')
            else:
                cnt_wait += 1
        
            if cnt_wait == args.patience:
                # print('Early stopping!')
                break
            if args.report and (epoch%5==0):
                print("Epoch {:05d}|loss_dgi {:.4f}|loss_class {:.4f}|acc_val {:.4f}"\
                      .format(epoch, loss_dgi.item(),loss_class.item(),acc_val))
        self.load_state_dict(torch.load('./model_save/'+args.dataset+'/best_model.pkl'))
    def train_ori_classifier_batch(self, g, args):
        g_iterator = ClusterIter(args.dataset, g, args.psize, args.batch_size, None)
        optimizer_class = torch.optim.Adam([{'params':self.encoder.parameters(),'lr':args.classifier_lr},
                                          {'params':self.classifier.parameters(),'lr':args.classifier_lr}],
                                         weight_decay=args.weight_decay)
        cnt_wait = 0
        best = 0
        for epoch in range(args.n_classifier_epochs):
            acc_list =  []
            for g_sub in g_iterator:
                if g_sub.ndata['train_mask'].sum()==0:
                    continue
                if args.gpu >= 0:
                    g_sub = g_sub.to(args.gpu)
                self.encoder.train()
                self.discriminator.train()
                optimizer_class.zero_grad()
                _, loss_class = self.forward(g_sub, True)
                loss = loss_class
                loss.backward()
                optimizer_class.step()
                
                acc_list.append(self.evaluate(g_sub, 'val_mask',args)[0])
            acc_val = torch.mean(torch.tensor(acc_list,dtype=float)).item()
            if acc_val > best:
                best = acc_val
                cnt_wait = 0
                torch.save(self.state_dict(), './model_save/'+args.dataset+'/best_model.pkl')
            else:
                cnt_wait += 1
        
            if cnt_wait == args.patience:
                # print('Early stopping!')
                break
            if args.report and (epoch%5==0):
                print("Epoch {:05d}|loss_class {:.4f}|acc_val {:.4f}"\
                      .format(epoch,loss_class.item(),acc_val))
        self.load_state_dict(torch.load('./model_save/'+args.dataset+'/best_model.pkl'))

    def evaluate(self, g, mask_name,args):
        if args.gpu >= 0:
            g = g.to(args.gpu)
        class_logits = self.get_class(g,args)
        mask = g.ndata[mask_name]
        if mask.sum()==0:
            return 0.
        logits_ = class_logits[mask]
        labels_ = g.ndata['label'][mask]
        _, indices = torch.max(logits_, dim=1)
        correct = torch.sum(indices == labels_)
        acc = correct.item() * 1.0 / len(labels_)
        miF1 = f1_score(labels_.cpu().numpy(),indices.cpu().numpy(), average='macro') 
        return acc, miF1
    def evaluate_batch(self, g, mask_name,args):
        g_iterator = ClusterIter(args.dataset, g, args.psize, args.batch_size, None)
        acc_list = []
        f1_list = []
        for g_sub in g_iterator:
            if g_sub.ndata[mask_name].sum()==0:
                    continue
            if args.gpu >= 0:
                g_sub = g_sub.to(args.gpu)
            class_logits = self.get_class(g_sub,args)
            mask = g_sub.ndata[mask_name]
            if mask.sum()==0:
                continue
            logits_ = class_logits[mask]
            labels_ = g_sub.ndata['label'][mask]
            _, indices = torch.max(logits_, dim=1)
            correct = torch.sum(indices == labels_)
            acc_ = correct.item() * 1.0 / len(labels_)
            miF1_ = f1_score(labels_.cpu().numpy(),indices.cpu().numpy(), average='macro')   
            f1_list.append(miF1_)
            acc_list.append(acc_)
        acc = torch.mean(torch.tensor(acc_list,dtype=float)).item()
        f1 = torch.mean(torch.tensor(f1_list,dtype=float)).item()
        return acc, f1
class Cluster(nn.Module):
    def __init__(self, args):
        super(Cluster, self).__init__()
        self.fc = nn.Linear(args.n_hidden, args.n_clusters)
        self.loss_link = LinkPredLoss()
        self.reset_parameters()
        if args.gpu >= 0:
            self.cuda()
    def reset_parameters(self):
        self.fc.reset_parameters()

    def forward(self, features):
        features = self.fc(features)
        return torch.softmax(features, dim=-1)
    def get_cluster(self, g_sub, args):
        if args.gpu >= 0:
            g_sub = g_sub.to(args.gpu)
        with torch.no_grad():
            cluster_logits = self.forward(g_sub.ndata['dgi_feat'])
        return cluster_logits.detach()
    def train_cluster(self, g_sub, args):#feature of g_sub need be dig feature
        optimizer_cluster = torch.optim.Adam(self.fc.parameters(),#[{'params':self.encoder.parameters()},{'params':self.cluster.parameters()}],
                                          lr=args.cluster_lr,
                                          weight_decay=args.weight_decay)
        cnt_wait = 0
        best = 1e9
        for epoch in range(args.n_cluster_epochs):
            if args.gpu >= 0:
                g_sub = g_sub.to(args.gpu)
            self.fc.train()
            cluster_logits = self.forward(g_sub.ndata['dgi_feat'])
            loss_cluster = self.loss_link(g_sub.edges(), cluster_logits)
            # self.encoder.train()
            optimizer_cluster.zero_grad()
            loss_cluster.backward()
            optimizer_cluster.step()
            
            loss = loss_cluster.item()
            if loss < best:
                best = loss
                cnt_wait = 0
                torch.save(self.state_dict(), './model_save/'+args.dataset+'/best_cluster.pkl')
            else:
                cnt_wait += 1
        
        self.load_state_dict(torch.load('./model_save/'+args.dataset+'/best_cluster.pkl'))



def get_PLCJ_train_mask(g, args):
    g_iterator = ClusterIter(args.dataset, g, args.psize, args.batch_size, None)
    cluster = Cluster(args)
    train_mask = torch.zeros(g.ndata['test_mask'].shape, dtype=bool)
    for g_sub in tqdm(g_iterator):
        cluster.train_cluster(g_sub, args)
        cluster_logits = cluster.get_cluster(g_sub, args).cpu()
        cluster_label = cluster_logits.max(1)[1]
        for i in range(args.n_clusters):
            # report_label(g_sub.ndata['label'][cluster_label==i],n_classes,False)
            center = g_sub.ndata['dgi_feat'][cluster_label==i].mean(0)
            can_choose_mask = (cluster_label==i) & ~g_sub.ndata['val_mask'] & ~g_sub.ndata['test_mask']
            if can_choose_mask.sum()==0:
                continue
            hs = g_sub.ndata['dgi_feat'][can_choose_mask]
            distance = (hs- center).norm(dim=1)
            pool_ids = g_sub.ndata['_ID'][can_choose_mask]
            chosen_id = pool_ids[distance.argmin()]
            train_mask[chosen_id] = True
    if train_mask.sum()<args.psize:
        can_choose_mask = ~train_mask & ~g.ndata['val_mask'] & ~g.ndata['test_mask']
        pool_ids = torch.arange(g.ndata['test_mask'].shape[0])[can_choose_mask]
        chosen_id = pool_ids[torch.randperm(pool_ids.shape[0])[:(args.psize-train_mask.sum())]]
        train_mask[chosen_id] = True
    return train_mask
















