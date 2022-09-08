import pandas as pd
import subprocess
import os
import numpy as np
import multiprocessing as mp
import sys
import argparse

# source = ""
# pseudo = "testp_text.tsv"
# testdf = "" 

# with open(pseudo) as f: 
#     before = f.read().replace("__gcd", "gcd")
# before = before.split("\n")[:-1]

df = None

packages = ["<iostream>", "<vector>", "<map>", "<set>", "<queue>", "<cmath>", "<iomanip>", "<stack>", "<numeric>", "<sstream>", "<cstdio>", "<climits>", "<cstring>", "<cstdlib>", "<algorithm>"]

original_compile_error = []
translation_compile_error = []
std_out = ""

def reconstruct_code(code_df): 
    global packages 
    code = ""
    for p in packages: 
        code += "#include " + p + "\n"
    code += "using namespace std;\n"
    for t in code_df.itertuples(): 
        code += "\t" * getattr(t, "indent") + getattr(t, "code") + "\n"
    return code

def reconstruct_all(): 
    for name, group in df.groupby(["subid", "workerid"]):     
        code = reconstruct_code(group)
        with open("code/" + str(name[0]) + ".cpp", "w") as f: 
            f.write(code)
        
def check_compilable(subid): 
    try: 
        subprocess.check_output("g++ -std=c++1y -o code_exe/" + subid + " code_rep/" + subid + ".cpp", shell = True, stderr=subprocess.PIPE)
        return 1
    except subprocess.CalledProcessError as e:
        translation_compile_error.append(e.stderr.decode("utf-8"))
        return 0
    
def check_compilable2(subid): 
    try: 
        subprocess.check_output("g++ -std=c++1y -o code_exe/" + subid + "_original code/" + subid + ".cpp", shell = True, stderr=subprocess.PIPE)
        return ""
    except subprocess.CalledProcessError as e:
        original_compile_error.append(e.stderr.decode("utf-8"))
        return subid
    
def parse_test_cases(probid, pp):
    tc = ""
    if pp == "all": 
        with open("chtc/testcases/" + probid + "/" + probid + "_testcases.txt") as f: 
            tc = f.read()
    else: 
        with open("chtc/testcases/" + probid + "/" + probid + "_testcases_public.txt") as f: 
            tc = f.read()
    tcs_in = []
    tcs_out = []
    tcs = tc.split("###ENDOUTPUT###")
    for i in tcs: 
        tt = i.split("###ENDINPUT###")
        if len(tt) == 2: 
            tcs_in.append(tt[0])
            tcs_out.append(tt[1])
    return (tcs_in, tcs_out)

def run_test_cases(tcs_in, tcs_out, exe, wid): 
    in_file_name = "code_exe/in_file" + str(wid) + ".txt"
    for i in range(len(tcs_in)): 
        with open(in_file_name, "w") as f_in: 
            f_in.write(tcs_in[i])

        with open(in_file_name, "r") as f_in: 
            try:
                #out = subprocess.run("ls code_exe",shell = True,  timeout=2, stdout=subprocess.PIPE)
                #print(out.stdout)
                out = subprocess.run("~/" + exe, shell = True, timeout = 2, stdin=f_in, stdout=subprocess.PIPE)
            except subprocess.TimeoutExpired:
                return 0
            try: 
                if out.stdout.decode("utf-8").strip() != tcs_out[i].strip():
                    return 0
            except: 
                return 0
    return 1  

def work(start, end, wid, source, testdf, beam):
    global df
    global std_out
    with open(source) as f: 
        after = f.read()
    after = after.split("\n")
    
    df = pd.read_csv(testdf, sep='\t')
    not_null = np.where(df['text'].notnull())[0]
   
    subprocess.run("mkdir code", shell=True)
    subprocess.run("mkdir code_exe", shell=True)
    subprocess.run("mkdir code_rep", shell=True)
    subprocess.run("mkdir outputs/" + source.split("/")[-1], shell=True)

    reconstruct_all()

    total_compilable = 0 
    total_passed = 0  

    src = source.split("/")[-1]
    out_file = "outputs/{}/{}_{}_".format(src, src, str(wid))
    with open(out_file + "output.txt", "w") as output_file:
        for i in range(start, end): 
            if i % 10 == 0: 
                t = "wid: {}, i: {}, total_compilable: {}, total_passed: {}".format(wid, i, total_compilable, total_passed)
                print(t)
                std_out += t + "\n"
                
            output_str = ""

            idx = not_null[i]
            subid = str(df.at[idx, "subid"])
            output_str += subid + ","

            probid = str(df.at[idx, "probid"])
            output_str += probid + ","
            
            # original compilable 
            if check_compilable2(subid) == "": 
                output_str += "True,"
            else: 
                output_str += "False,,,,,\n"
                output_file.write(output_str)
                continue

            tcs_in, tcs_out = parse_test_cases(probid, "all")

            # original passed
            if run_test_cases(tcs_in, tcs_out, "code_exe/" + subid + "_original", wid) == 1: 
                output_str += "True,"
            else: 
                output_str += "False,,,,\n"
                output_file.write(output_str)
                continue

            rep_line_idx = df.at[idx, "line"] + len(packages) + 1
            rep_line_indent = df.at[idx, "indent"]

            with open("code/" + subid + ".cpp") as f:
                original_code = f.read()
            rep_code = original_code.split("\n")

            compilable = 0 
            passed = 0 
            for j in range(beam):
                rep_line = "\t" * rep_line_indent + after[i * 5 + j]
                rep_code[rep_line_idx] = rep_line
                rep_code_join = "\n".join(rep_code)

                with open("code_rep/" + subid + ".cpp", "w") as f: 
                    f.write(rep_code_join)

                res = check_compilable(subid)

                if res == 1: 
                    compilable += 1
                    total_compilable += 1
                    pass_res = run_test_cases(tcs_in, tcs_out, "code_exe/" + subid, wid)
                    if pass_res == 1:
                        passed += 1 
                        total_passed += 1  

                if j == 0: 
                    if compilable == 1: 
                        output_str += "True,"
                    else: 
                        output_str += "False,"
                    if passed == 1: 
                        output_str += "True,"
                    else: 
                        output_str += "False,"

            if compilable > 0: 
                output_str += "True,"
            else: 
                output_str += "False,\n"
                output_file.write(output_str)
                continue 

            if passed > 0:
                output_str += "True\n"
            else: 
                output_str += "False\n"
            output_file.write(output_str)
    with open(out_file + "stdout.txt", "w") as f:
        f.write(std_out + "\n" + str(total_compilable) + ", " + str(total_passed))
    with open(out_file + "original_compile_error.txt", "w") as f:
        f.write("\n".join(original_compile_error))
    with open(out_file + "translation_compile_error.txt", "w") as f:
        f.write("\n".join(translation_compile_error))
    return (total_compilable, total_passed)

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', action="store", type=int)
    parser.add_argument('--scale', action="store", type=int)
    parser.add_argument('--source', action="store")
    parser.add_argument('--testdf', action="store")
    parser.add_argument('--beam', action="store", type=int)
    
    args = parser.parse_args()
    
    start = args.start
    scale = args.scale
    
    if (start + 1) * scale > 15183: 
        print(work(start * scale, 15183, start, args.source, args.testdf, args.beam))
    else:   
        print(work(start * scale, (start + 1) * scale, start, args.source, args.testdf, args.beam))

if __name__=="__main__":
#     parallel programming, not necessary on GPU server
#     num_workers = 2
#     args = []
#     for i in range(num_workers): 
#         args.append((i * scale, (i + 1) * scale, i))
#     with mp.Pool(processes=num_workers) as pool:
#         pool.starmap(work, args)
#     print(work(i * scale, (i + 1) * scale, i))
    main()
