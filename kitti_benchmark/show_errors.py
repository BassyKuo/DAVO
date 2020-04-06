import os
import argparse
import glob
import numpy as np

def argsparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("err_dir", 
            help="Directory of error files generated from test_odometry/eval_odometry.    "
                 "ex: results/pretrain-model-258000/errors/")
    parser.add_argument("-seq", nargs='+', type=int,
            choices=range(0,11),
            help="Choose specitic sequences to show.   "
                 "ex: -seq 0 1 2 10")
    parser.add_argument("--output_to_file", "-o", action='store_true',
            help="Output to file.")
    return parser.parse_args()

def main(args):
    err_files = []
    if args.seq:
        for s in args.seq:
            seq_file = os.path.join(args.err_dir, '%.2d.txt' % s)
            err_files.append(seq_file)
    else:
        err_files = sorted(glob.glob(os.path.join(args.err_dir, '*.txt')))

    file_name = os.path.dirname(args.err_dir + '/').split('/')[-2] + '.csv'
    titles = ["seq", "transErr", "rotErr"]
    writer = { name:[] for name in title }

    for err_file in err_files:
        seq_name = os.path.splitext(os.path.basename(err_file))[0]
        with open(err_file, 'r') as f:
            row = f.read().splitlines()
        results = [np.array([float(x) for x in line.split(' ')]) for line in row]
        results = np.stack(results, axis=0)
        rotErr_mean = results[:,1].mean() * 57.3                                    # "* 57.3" = Radian -> Degree. by kitti_benchmark devkit (test_odometry)
        transErr_mean = results[:,2].mean() * 100.                                  # "*100" = unit -> %.          by kitti_benchmark devkit (test_odometry)
        writer["seq"].append(seq_name)
        writer["rotErr"].append(rotErr_mean)
        writer["transErr"].append(transErr_mean)

    writer["seq"].append("ave")
    writer["rotErr"].append(np.mean(writer["rotErr"]))
    writer["transErr"].append(np.mean(writer["transErr"]))

    if args.output_to_file:
        # Sequence Error
        with open(file_name, 'w') as f:
            for idx in range(len(writer['seq'])):
                f.write("%s,%s," % (str(writer["transErr"][idx]), str(writer["rotErr"][idx])))
    else:
        print (','.join(titles))
        for idx in range(len(writer['seq'])):
            print(str(writer["seq"][idx]), str(writer["transErr"][idx]), str(writer["rotErr"][idx]))



if __name__ == '__main__':
    main(argsparser())
