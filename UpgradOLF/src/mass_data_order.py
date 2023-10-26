import os
import numpy as np


def mass_data_order(orignal, new_name, max_order_len):
    base_dir = os.path.dirname(orignal)
    org_txt = open(orignal, 'r')
    seq_dict = {}
    org_lines = org_txt.readlines()
    for line in org_lines:
        line = line.replace("\n", "")
        seq = os.path.dirname(line)
        if seq not in seq_dict:
            seq_dict[seq] = [line]
        else:
            seq_dict[seq].append((line))
    org_txt.close()
    # sorted
    new_dict = {}
    for seq in seq_dict:
        new_dict[seq] = sorted(seq_dict[seq])

    # write
    with open(new_name, 'w') as file:
        availables = find_available_seq(new_dict, max_order_len)
        while len(availables) > 0:
            ind = np.random.randint(len(availables))
            key = availables[ind]
            for i in range(max_order_len):
                new_line = new_dict[key].pop(0)
                file.write(new_line + "\n")
            availables = find_available_seq(new_dict, max_order_len)

        # less
        values = []
        for vals in new_dict.values():
            for val in vals:
                values.append(val)
        np.random.shuffle(values)
        for value in values:
            file.write(value + "\n")


def find_available_seq(seq_dict, max_len):
    L = []
    for key in seq_dict:
        if len(seq_dict[key]) >= max_len:
            L.append(key)

    return L


if __name__ == '__main__':
    orig = "../src/data/mot17.half"
    new_name = "../src/data/mot17_mass_order.half"

    mass_data_order(orig, new_name, 10)

    orig = "../src/data/mot20.half"
    new_name = "../src/data/mot20_mass_order.half"

    mass_data_order(orig, new_name, 10)