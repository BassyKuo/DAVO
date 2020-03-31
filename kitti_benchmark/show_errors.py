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
    hfile_name = os.path.dirname(args.err_dir + '/').split('/')[-2] + '.csv-h-RMSE'
    titles = ["seq", "transMean", "rotMean", "transRMSE", "rotRMSE"]
    writer = {
            "seq" : [],
            "transMean" : [],
            "rotMean" : [],
            "transRMSE" : [],
            "rotRMSE" : []
            }

    #print ("%5s  %-9s  %-8s" % ("seq", "transMean", "rotMean"))
    #print ("===========================")
    for err_file in err_files:
        seq_name = os.path.splitext(os.path.basename(err_file))[0]
        with open(err_file, 'r') as f:
            row = f.read().splitlines()
        results = [np.array([float(x) for x in line.split(' ')]) for line in row]
        results = np.stack(results, axis=0)
        rotErr_mean = results[:,1].mean() * 57.3                                    # "* 57.3" = Radian -> Degree. by kitti_benchmark devkit (test_odometry)
        transErr_mean = results[:,2].mean() * 100.                                  # "*100" = unit -> %.          by kitti_benchmark devkit (test_odometry)
        rotErr_rmse = np.sqrt( np.mean( np.power(results[:,1], 2) ) ) * 57.3        # "* 57.3" = Radian -> Degree. by kitti_benchmark devkit (test_odometry)
        transErr_rmse = np.sqrt( np.mean( np.power(results[:,2], 2) ) ) * 100.      # "*100" = unit -> %.          by kitti_benchmark devkit (test_odometry)
        #print ("%5s   %.6f  %.6f" % (seq_name, transErr_mean, rotErr_mean))
        writer["seq"].append(seq_name)
        writer["rotMean"].append(rotErr_mean)
        writer["transMean"].append(transErr_mean)
        writer["rotRMSE"].append(rotErr_rmse)
        writer["transRMSE"].append(transErr_rmse)
    #print ("---------------------------")
    #print ("%5s   %.6f  %.6f" % ("all", np.mean(transErr_means), np.mean(rotErr_means)))

    writer["seq"].append("ave")
    writer["rotMean"].append(np.mean(writer["rotMean"]))
    writer["transMean"].append(np.mean(writer["transMean"]))
    writer["rotRMSE"].append(np.mean(writer["rotRMSE"]))
    writer["transRMSE"].append(np.mean(writer["transRMSE"]))

    if args.output_to_file:
        #with open(file_name, 'w') as f:
            #f.write(','.join(titles))
            #for idx in range(len(writer['seq'])):
                #f.write("\n%s,%s,%s" % (str(writer["seq"][idx]), str(writer["transMean"][idx]), str(writer["rotMean"][idx]), str(writer["transRMSE"][idx]), str(writer["rotRMSE"][idx])))
        # RMS Error
        with open(hfile_name, 'w') as f:
            for idx in range(len(writer['seq'])-1):
                f.write("%s,%s," % (str(writer["transRMSE"][idx]), str(writer["rotRMSE"][idx])))
            idx+=1
            f.write("%s,%s\n" % (str(writer["transRMSE"][idx]), str(writer["rotRMSE"][idx])))
        # Averaged Error
        with open(file_name, 'w') as f:
            for idx in range(len(writer['seq'])-1):
                f.write("%s,%s," % (str(writer["transMean"][idx]), str(writer["rotMean"][idx])))
            idx+=1
            f.write("%s,%s\n" % (str(writer["transMean"][idx]), str(writer["rotMean"][idx])))
    else:
        print (','.join(titles))
        for idx in range(len(writer['seq'])):
            print(str(writer["seq"][idx]), str(writer["transMean"][idx]), str(writer["rotMean"][idx]), str(writer["transRMSE"][idx]), str(writer["rotRMSE"][idx]))



if __name__ == '__main__':
    main(argsparser())
