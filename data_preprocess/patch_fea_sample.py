# %%
import pickle
import torch
feature_file = './data/Feat_result/TCGA_BLCA_input/BLCA_UNI_features.pkl'
id_file = './data/Feat_result/TCGA_BLCA_input/blca_five_folds.pkl'
with open(feature_file,'rb') as f:
    features = pickle.load(f)

with open(id_file,'rb') as f:
    idfold = pickle.load(f)

patch_fold = {}
torch.manual_seed(98)
for fold in range(5):
    print(fold)
    patch_500_all = []
    for id in idfold['train_' + str(fold)]:
        # print(id)
        fea = features[id]
        indices = torch.randperm(fea.size(0))[:1500] 
        x_random_500 = fea[indices]
        if x_random_500.shape[0] < 1500:
            row_mean = torch.mean(x_random_500, dim=0, keepdim=True)
            padding = row_mean.repeat(1500-x_random_500.shape[0], 1)
            x_random_500 = torch.cat([x_random_500, padding], dim=0)
        patch_500_all.append(x_random_500)
        print(id)
    x_concatenated = torch.stack(patch_500_all)
    print(x_concatenated.shape)
    patch_fold['fold' + str(fold)] = x_concatenated

with open('./data/Feat_result/TCGA_BLCA_input/patch_1500_all.pkl','wb') as f:
    pickle.dump(patch_fold,f)