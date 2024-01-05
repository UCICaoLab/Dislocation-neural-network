# This code is developed to read barriers from NEB calculations
import os
import pathlib

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
    
def main():
    current_path = pathlib.Path(__file__).parent.absolute()
    NEB_files = [f for f in os.listdir(current_path) if f.startswith("log.neb.log")]
    NEB_files.sort(key=lambda x:int(x.split('.')[-1]))
    last_lines = []
    for input_file in NEB_files:
        last_lines.append(get_last_line(input_file).rstrip('\n').split()[6])
    output_file = os.path.join(__location__, "barrier.txt")
    o_file = open(output_file, 'w+')
    for word in last_lines:
        o_file.write(word + ",")

def get_last_line(file):
    last_line = ""
    with open(os.path.join(__location__, file)) as f:
        for line in f:
            pass
        last_line = line
    return last_line

if __name__ == "__main__":
    # execute only if run as a script
    main()
