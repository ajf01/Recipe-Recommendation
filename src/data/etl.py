# Copyright (c) 2017 NVIDIA Corporation
from os import listdir, path, makedirs
import random
import sys

def save_data_to_file(data, filename):
    with open(filename, 'w') as out:
        for userId in data:
            for record in data[userId]:
                out.write("{}\t{}\t{}\n".format(userId, record[0], record[1]))

def main(configs):
    folder = configs['original_data']
    out_folder = configs['output_location']
    # create necessary folders:
    makedirs('data', exist_ok=True)
    for output_dir in [(out_folder + f) for f in [
        "/N3M_TRAIN", "/N3M_VALID", "/N3M_TEST", "/N6M_TRAIN",
        "/N6M_VALID", "/N6M_TEST", "/N1Y_TRAIN", "/N1Y_VALID",
        "/N1Y_TEST", "/NF_TRAIN", "/NF_VALID", "/NF_TEST"]]:
        makedirs(output_dir, exist_ok=True)

    text_files = [path.join(folder, f)
                    for f in listdir(folder)
                    if path.isfile(path.join(folder, f)) and ('.txt' in f)]


    for text_file in text_files:


        with open(text_file, 'r') as f:
            print("Processing: {}".format(text_file))
            lines = f.readlines()
            item = int(lines[0][:-2]) # remove newline and :
            if not item in item2id_map:
                item2id_map[item] = itemId
                itemId += 1

        for rating in lines[1:]:
            parts = rating.strip().split(",")
            user = int(parts[0])
            if not user in user2id_map:
                user2id_map[user] = userId
                userId += 1
            rating = float(parts[1])
            ts = int(time.mktime(datetime.datetime.strptime(parts[2],"%Y-%m-%d").timetuple()))
            if user2id_map[user] not in all_data:
                all_data[user2id_map[user]] = []
            all_data[user2id_map[user]].append((item2id_map[item], rating, ts))
        
    print("STATS FOR ALL INPUT DATA")
    print_stats(all_data)

    # Netflix full
    (nf_train, nf_valid, nf_test) = create_NETFLIX_data_timesplit(all_data,
                                                                "1999-12-01",
                                                                "2005-11-30",
                                                                "2005-12-01",
                                                                "2005-12-31")
    print("Netflix full train")
    print_stats(nf_train)
    save_data_to_file(nf_train, out_folder + "/NF_TRAIN/nf.train.txt")
    print("Netflix full valid")
    print_stats(nf_valid)
    save_data_to_file(nf_valid, out_folder + "/NF_VALID/nf.valid.txt")
    print("Netflix full test")
    print_stats(nf_test)
    save_data_to_file(nf_test, out_folder + "/NF_TEST/nf.test.txt")


    (n3m_train, n3m_valid, n3m_test) = create_NETFLIX_data_timesplit(all_data,
                                                                    "2005-09-01",
                                                                    "2005-11-30",
                                                                    "2005-12-01",
                                                                    "2005-12-31")
    print("Netflix 3m train")
    print_stats(n3m_train)
    save_data_to_file(n3m_train, out_folder+"/N3M_TRAIN/n3m.train.txt")
    print("Netflix 3m valid")
    print_stats(n3m_valid)
    save_data_to_file(n3m_valid, out_folder + "/N3M_VALID/n3m.valid.txt")
    print("Netflix 3m test")
    print_stats(n3m_test)
    save_data_to_file(n3m_test, out_folder + "/N3M_TEST/n3m.test.txt")

    (n6m_train, n6m_valid, n6m_test) = create_NETFLIX_data_timesplit(all_data,
                                                                    "2005-06-01",
                                                                    "2005-11-30",
                                                                    "2005-12-01",
                                                                    "2005-12-31")
    print("Netflix 6m train")
    print_stats(n6m_train)
    save_data_to_file(n6m_train, out_folder+"/N6M_TRAIN/n6m.train.txt")
    print("Netflix 6m valid")
    print_stats(n6m_valid)
    save_data_to_file(n6m_valid, out_folder + "/N6M_VALID/n6m.valid.txt")
    print("Netflix 6m test")
    print_stats(n6m_test)
    save_data_to_file(n6m_test, out_folder + "/N6M_TEST/n6m.test.txt")

    # Netflix 1 year
    (n1y_train, n1y_valid, n1y_test) = create_NETFLIX_data_timesplit(all_data,
                                                                    "2004-06-01",
                                                                    "2005-05-31",
                                                                    "2005-06-01",
                                                                    "2005-06-30")
    print("Netflix 1y train")
    print_stats(n1y_train)
    save_data_to_file(n1y_train, out_folder + "/N1Y_TRAIN/n1y.train.txt")
    print("Netflix 1y valid")
    print_stats(n1y_valid)
    save_data_to_file(n1y_valid, out_folder + "/N1Y_VALID/n1y.valid.txt")
    print("Netflix 1y test")
    print_stats(n1y_test)
    save_data_to_file(n1y_test, out_folder + "/N1Y_TEST/n1y.test.txt")


if __name__ == "__main__":
    main(sys.argv)
