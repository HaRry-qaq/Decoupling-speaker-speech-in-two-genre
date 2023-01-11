#!/usr/bin/env python
# encoding: utf-8

import os
import argparse
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--cnceleb_root', help='cnceleb dir', type=str, default="CN-Celeb")
    parser.add_argument('--dst_trl_path', help='output trial path', type=str, default="new.trials")
    parser.add_argument('--apply_vad', action='store_true', default=False)
    args = parser.parse_args()
    # cnceleb_root = '/work8/zhouzy/baseline/sunine-master/CN-Celeb'
    # dst_trl_path = '/work8/zhouzy/dgt/sunine_multi_eer/sunine/egs/cnceleb/v1/local'
    
    enroll_lst_path = os.path.join(args.cnceleb_root, "eval/lists/enroll.lst")
    raw_trl_path = os.path.join(args.cnceleb_root, "eval/lists/trials.lst")
    test_trl_path = os.path.join(args.cnceleb_root, "eval/lists/test.lst")

    spk2wav_mapping = {}
    enroll_lst = np.loadtxt(enroll_lst_path, str)
    for item in enroll_lst:
        spk2wav_mapping[item[0]] = item[1]
    trials = np.loadtxt(raw_trl_path, str)
    test_trials = np.loadtxt(test_trl_path, str)
    gener = []
    for item in test_trials:
        gener.append(item.split('/')[1].split('-')[1])
    gener = list(set(gener))
    # print(gener)
    for i in range(len(gener)):
        # print(os.path.join(dst_trl_path, gener[i]))
        with open(os.path.join(args.dst_trl_path, gener[i]+'.lst'), "w") as f:
            # print(os.path.join(args.dst_trl_path, gener[i]+'.lst'))
            for item in trials:
                # print(item[1].split('/')[1].split('-')[1])
                if item[1].split('/')[1].split('-')[1] == gener[i]:
                    enroll_path = os.path.join(args.cnceleb_root, "eval", spk2wav_mapping[item[0]])
                    test_path = os.path.join(args.cnceleb_root, "eval", item[1])
                    if args.apply_vad:
                        enroll_path = enroll_path.strip("*.wav") + ".vad"
                        test_path = test_path.strip("*.wav") + ".vad"
                    label = item[2]
                    f.write("{} {} {}\n".format(label, enroll_path, test_path))


    with open(args.dst_trl_path + '/CNC-Eval-Core.lst', "w") as f:
        for item in trials:
            # print('item:',item)
            enroll_path = os.path.join(args.cnceleb_root, "eval", spk2wav_mapping[item[0]])
            test_path = os.path.join(args.cnceleb_root, "eval", item[1])
            if args.apply_vad:
                enroll_path = enroll_path.strip("*.wav") + ".vad"
                test_path = test_path.strip("*.wav") + ".vad"
            label = item[2]
            f.write("{} {} {}\n".format(label, enroll_path, test_path))
