import json
import os
import pathlib
import shutil

DATA_FILE_PATH = r"C:\Users\Anchit Jain\Downloads\arxiv_dataset\arxiv-metadata-oai-snapshot.json"
"""
def get_metadata(data_file):
    with open(data_file, 'r') as f:
        for line in f:
            yield line
i=0
j=0
license_dic = {}
for data in get_metadata(DATA_FILE_PATH):
    data_dic = json.loads(data)
    if data_dic['license']:
        if data_dic['license'] in license_dic:
            license_dic[data_dic['license']] += 1
        else:
            license_dic[data_dic['license']] = 1

print(license_dic)
"""
# License Summary:
# Total: 2134841
# 'http://arxiv.org/licenses/nonexclusive-distrib/1.0/': 1457448, 
# 'http://creativecommons.org/licenses/by-nc-nd/4.0/': 21653, 
# 'http://creativecommons.org/licenses/publicdomain/': 2478, 
# 'http://creativecommons.org/licenses/by-nc-sa/3.0/': 5878, 
# 'http://creativecommons.org/licenses/by/3.0/': 7921, 
# 'http://creativecommons.org/licenses/by/4.0/': 146774, 
# 'http://creativecommons.org/licenses/by-nc-sa/4.0/': 21661, 
# 'http://creativecommons.org/publicdomain/zero/1.0/': 9001, 
# 'http://creativecommons.org/licenses/by-sa/4.0/': 9146

TEST_PATH = r"C:\Users\Anchit Jain\ML_projects\CLIP_data\scicap_data\scicap_data\SciCap-Caption-All\train\1910.05758v3-Figure8-1.json"
TEST_PATH_2 = r"C:\Users\Anchit Jain\ML_projects\CLIP_data\scicap_data\scicap_data\SciCap-Caption-All\train"

def json_to_dic(filepath):
    """Reads a .json file and returns a dictionary with 
    the contents of the json file"""
    f = open(filepath, 'r')
    json_dic = json.load(f)
    f.close()
    return json_dic

def json_from_dir_generator(dir_path):
    """Creates a generator that yields filenames
    target directory"""
    p = pathlib.Path(dir_path)
    for filename in p.iterdir():
        yield filename

"""
#print(json_to_dic(TEST_PATH)['contains-subfigure'])
j = 0
for i in json_from_dir_generator(TEST_PATH_2):
    print(i['figure-ID'])
    j +=1
    if j>10:
        break
"""
def extract_caption_from_dic(json_dic, txt_file_path):
    """given a dictionary extracted from a json file,
    extracts and writes the caption to a .txt file of the specified name"""
    caption = json_dic["1-lowercase-and-token-and-remove-figure-index"]['caption']
    f = open(txt_file_path, 'w', encoding='utf-8')
    #print(caption)
    f.write(caption)
    f.close()

def main(source_caption_dir, source_figure_dir, target_data_dir, dataset_size = 100):
    data_counter = 0
    for filename in json_from_dir_generator(source_figure_dir):
        figure_name = str(filename.name)[:-4]
        #print(figure_name)
        json_dic = json_to_dic(os.path.join(source_caption_dir, figure_name + '.json'))
        #extract_caption_from_dic(json_dic, os.path.join(target_data_dir, figure_name + '.txt'))
        shutil.copy(os.path.join(source_figure_dir, figure_name + '.png'), os.path.join(target_data_dir, figure_name + '.png'))
        data_counter += 1
        if data_counter%100 == 0:
            print(data_counter)
        if data_counter>=dataset_size:
            break

if __name__ == "__main__":
    SOURCE_CAPTION_DIR = r"C:\Users\Anchit Jain\ML_projects\CLIP_data\scicap_data\scicap_data\SciCap-Caption-All\test"
    SOURCE_FIGURE_DIR = r"C:\Users\Anchit Jain\ML_projects\CLIP_data\scicap_data\scicap_data\SciCap-No-Subfig-Img\test"
    TARGET_DIR = r"C:\Users\Anchit Jain\ML_projects\CLIP_data\scicap_data\scicap_test_data"
    main(SOURCE_CAPTION_DIR, SOURCE_FIGURE_DIR, TARGET_DIR , dataset_size=1000)