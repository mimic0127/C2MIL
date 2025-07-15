# %%
import argparse
import torch
import os
import PIL.Image as Image
import timm
import sys
import torchvision.transforms as T
sys.path.append('./causal_double_A800/data_preprocess/CONCH')
from resnet import resnet50_baseline
import glob
from conch.open_clip_custom import create_model_from_pretrained
from torch.utils.data import Dataset
parser = argparse.ArgumentParser(description='thumbnail')
parser.add_argument('--data_dir', default='', type=str)
parser.add_argument('--device', default='1', type=str)
parser.add_argument('--num_worker', default=0, type=int)
parser.add_argument('--batch_size_v', default=1, type=int)
parser.add_argument('--model', default='UNI', type=str)
parser.add_argument('--img_resize', default=224, type=int)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.device
# %%
if args.model == 'ResNet':
    fea_extractor = resnet50_baseline(pretrained=True).cuda()
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_transform = T.Compose([
        #RandomCrop(224),
        T.ToTensor(),
        normalize,
    ])
if args.model == 'UNI':
    fea_extractor = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
    )
    fea_extractor = fea_extractor.cuda()
    fea_extractor.load_state_dict(torch.load(os.path.join("/cache/cenmin/pretrained_model/UNI_weights/", "pytorch_model.bin"), map_location="cpu"), strict=True)
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_transform = T.Compose([
        #RandomCrop(224),
        T.ToTensor(),
        normalize,
    ])
if args.model == 'conch':
    fea_extractor,test_transform = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path="/cache/cenmin/pretrained_model/conch/pytorch_model.bin") 
    fea_extractor = fea_extractor.cuda()

thumbnail_root = './data/Thumnail/TCGA_BLCA'
thumbfea_root = './data/thumbnail_fea/TCGA_BLCA_' + args.model
if not os.path.exists(thumbfea_root):
    os.makedirs(thumbfea_root)

class thumbnail_datasets(Dataset):
    def __init__(self, thumb_root, transform=None, img_resize=224):
        self.img_resize = img_resize
        self.thumb_root = thumb_root
        self.transform = transform
        self.img_list = glob.glob(self.thumb_root + '/*')
    
    def __getitem__(self, index):
        file_path = self.img_list[index]
        id = file_path[file_path.rindex('/')+1:-4]
        image = Image.open(file_path)
        try:
            timage = image.convert('RGB')
        except OSError as e:
            print(f"Error loading image: {e},print{id}")
        timage = timage.resize((self.img_resize,self.img_resize))
        if self.transform is not None:
            timage = self.transform(timage)

        return {'image':timage,'id':id}      

    def __len__(self):
        return len(self.img_list)   

def extract_save_features(extractor,loader,params,save_path=''):
    extractor.eval()
    for idx, batchdata in enumerate(loader):
        samples = batchdata['image'].cuda()
        id = batchdata['id'][0]
        # if id != "TCGA-C8-A27":
        #     continue
        print(id)
        if os.path.exists(thumbfea_root+'/' +id + '.pt') or id == '':
            print(id,'exist!!!')
            continue
        if args.model == 'conch':
            feat = extractor.encode_image(samples, proj_contrast=False, normalize=False)
            # print(feat)
        else:
            feat = extractor(samples).cpu()
        torch.save(feat,thumbfea_root+'/' +id + '.pt')

all_dataset = thumbnail_datasets(thumbnail_root,test_transform)
all_loader = torch.utils.data.DataLoader(
            all_dataset, batch_size=args.batch_size_v, shuffle=False,
            num_workers=args.num_worker, pin_memory=True)
# %%
print('============== start extract features ===========')
extract_save_features(extractor=fea_extractor, loader=all_loader, params=args,
                            save_path=thumbfea_root)
