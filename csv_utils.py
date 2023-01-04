import json
import os
import pathlib
import pandas as pd


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

def list_to_csv(list, csv_path, columns=["figure_index", "figure_name", "fig_path", "caption"]):
    df = pd.DataFrame(list, columns=columns)
    df.to_csv(csv_path)


def main(source_fig_dir, source_cap_dir, csv_path):
    csv_list = []
    data_counter = 0
    for filename in json_from_dir_generator(source_fig_dir):
        figure_name = str(filename.name)[:-4]
        json_dic = json_to_dic(os.path.join(source_cap_dir, figure_name + '.json'))
        caption = json_dic["1-lowercase-and-token-and-remove-figure-index"]['caption']
        #caption = caption.encode(encoding='utf-8')
        csv_list.append([data_counter, figure_name, os.path.join(source_fig_dir, filename), caption])
        data_counter += 1
    list_to_csv(csv_list, csv_path)

if __name__ == "__main__":
    SOURCE_FIG_DIR = r"C:\Users\Anchit Jain\ML_projects\CLIP_data\scicap_data\scicap_test_data"
    SOURCE_CAP_DIR = r"C:\Users\Anchit Jain\ML_projects\CLIP_data\scicap_data\scicap_data\SciCap-Caption-All\test"
    CSV_PATH = r"C:\Users\Anchit Jain\ML_projects\CLIP_data\scicap_data\scicap_test_data\test.csv"
    
    main(SOURCE_FIG_DIR, SOURCE_CAP_DIR, CSV_PATH)

    




