# %%
import os
import pickle
import glob
id_label_path = './data/Feat_result/TCGA_BLCA_input/blca_sur_and_time.pkl'
with open(id_label_path,'rb') as file:
    id_label = pickle.load(file)
print(id_label)
patient_list = [i + '-01Z-00-DX1' for i in id_label.keys()]
# %%
slide_root = '/SLIDE_ROOT/BLCA'
slide_list = glob.glob(slide_root + '/*.svs')
slide_list = [i for i in slide_list if i[i.rindex('/') + 1:i.index('.')] in patient_list]
print(slide_list)
# %%
import openslide
from tqdm import tqdm 
if not os.path.exists('./data/Thumnail/TCGA_BLCA'):
    os.makedirs('./data/Thumnail/TCGA_BLCA')
for svs_path in tqdm(slide_list):
    print(svs_path)
    slide = openslide.OpenSlide(svs_path) 
    level0_width, level0_height = slide.level_dimensions[0]
    thumbnail = slide.get_thumbnail((level0_width//30, level0_height//30))
    thumbnail.save('./data/Thumnail/TCGA_BLCA/'+svs_path[svs_path.rindex('/') + 1:svs_path.rindex('/') +13] + '.png')