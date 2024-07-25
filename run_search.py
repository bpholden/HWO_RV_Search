import argparse
import glob
import os
import re
import sys

import rvsearch
from rvsearch import search, utils

def parse_args():

    parser = argparse.ArgumentParser(description="Run rvsearch on a csv file in a given directory")
    parser.add_argument('-j','--jump', action='store_true', help='Format of csv data in Jump format')
    parser.add_argument('-m','--minerv', action='store_true', help='Format of csv data in Minerva format')
    parser.add_argument('-p','--min_per', type=float, default=10, help='Minimum period for search')
    parser.add_argument('-c','--num_cpu', type=int, default=10, help='Number of CPUs for search.')
    parser.add_argument('datapath', type=str, help='Directory of the data')

    args = parser.parse_args()

    return args

def parse_jump(fn):


    # Jump files have a bjd instead of a jd column
    # the time column is a UT ISOT time stamp, not a JD
    # and by default rvsearch looks for a time column first
    # and assumes that the values are JD
    # so we get rid of it and rename the bjd
    
    vel_data = utils.read_from_csv(fn)
    vel_data['jd'] = vel_data['bjd']
    vel_data.pop('time')

    return vel_data

def main():

    args = parse_args()

    fns = glob.glob(os.path.join(args.datapath,"*csv"))
    print(fns[0])

    if args.jump:
        vel_data = parse_jump(fns[0])

    searcher = search.Search(vel_data, starname=args.datapath,
                                 min_per=args.min_per,
                                 workers=args.num_cpu,
                                 mcmc=True, verbose=True)
    searcher.run_search()
    searcher.save(os.path.join(args.datapath,"post_final.pkl"))

if __name__ == "__main__":
    main()
