import json
DATA_FILE_PATH = r"C:\Users\Anchit Jain\Downloads\arxiv_dataset\arxiv-metadata-oai-snapshot.json"
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