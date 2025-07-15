import os
import pickle
import torch
dataset_name = 'TCGA_BLCA'
id_file = f"./data/Feat_result/{dataset_name}_input/blca_five_folds.pkl"

with open(id_file,'rb') as f:
    idfold = pickle.load(f)

thumbnail_fea_root = './data/thumbnail_fea'
fea_type = 'UNI'
thumbnail_fold = {}
for fold in range(5):
    print(fold)
    thumb_fea = []
    print(len(idfold['train_'+str(fold)]))
    print(len(idfold['val_'+str(fold)]))
    for id in idfold['train_' + str(fold)]:
        thumb_path = os.path.join(thumbnail_fea_root,dataset_name + '_' + fea_type, id + '.pt')
        fea = torch.load(thumb_path,map_location='cpu')
        print(fea.shape)
        thumb_fea.append(fea)
    thumb_fea_all = torch.concat(thumb_fea)
    print(thumb_fea_all.shape)
    thumbnail_fold['fold' + str(fold)] = thumb_fea_all

with open(f'./data/Feat_result/{dataset_name}_input/thumbnail_all_'+fea_type + '.pkl','wb') as f:
    pickle.dump(thumbnail_fold,f)