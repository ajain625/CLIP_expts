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
# i, j = 2134841 224512