# import numpy
import argparse
import statistics as stats

def extract_accuracy(input_file, output_file, n=3):
    print(input_file)
    with open(input_file, 'r') as fr:
        lines=fr.readlines()

        dev_accs=[]
        test_accs=[]

        for line in lines:
            # print(line)
            line = line.strip()
            if line.startswith("best dev acc"):
                dev=float(line.split("\t")[1])
                dev_accs.append(dev)
            if line.startswith("final test acc"):
                test=float(line.split("\t")[1])
                test_accs.append(test)

        accs = { x:y for x,y in zip(dev_accs,test_accs)}
        accs = [(k,v) for k,v in sorted(accs.items())]

        for acc in accs:
            print(acc)

        accs_topn = accs[-n:]
        print("\n-----------") 
        for acc in accs_topn:
            print(acc)

        compute_mean_stdev(accs_topn, output_file, index=input_file, n=n)
    return accs_topn

def compute_mean_stdev(accs_topn, output_file, index, n):
    dev_accs = [x[0] for x in accs_topn]
    dev_miu = stats.mean(dev_accs)
    dev_rho = stats.stdev(dev_accs)

    test_accs = [x[1] for x in accs_topn]
    test_miu = stats.mean(test_accs)
    test_rho = stats.stdev(test_accs)
     
    dev_out = "{:.2f} (±{:.2f})".format(dev_miu*100, dev_rho*100)
    test_out = "{:.2f} (±{:.2f})\n".format(test_miu*100, test_rho*100)

    with open(output_file, 'a+') as fa:
        fa.write(index+f" top{n} dev_acc test_acc\n")
        fa.write("{}\t{}\n".format(dev_out, test_out))

    print(dev_out, test_out)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str,)
    parser.add_argument("--output_file", type=str,)

    args = parser.parse_args()
    input_file="slurm-22613565.out"
    output_file="./log/auto_results.md"

    args.input_file=f"slurm-{args.input_file}.out"

    extract_accuracy(args.input_file, args.output_file)
# def input_files():