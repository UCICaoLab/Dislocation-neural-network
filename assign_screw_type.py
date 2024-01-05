# This code is to randomly assign atom types to an initial lmp data file

import random
import os
import pathlib
import shutil
current_path = pathlib.Path(__file__).parent.absolute()
old_file = os.path.join(current_path, 'initial_screw.lmp')
def main():
	p1=0.6
	p2=0.3
	p3=round(1-p1-p2,3)
	total_number=960 
	type1=int(p1*total_number)
	type2=int(p2*total_number)
	type3=int(total_number-type1-type2)
	__location__ = os.path.realpath(
		os.path.join(os.getcwd(), os.path.dirname(__file__)))

	type_list=[]
	for i in range(type1):
		type_list.append(1)
	for i in range (type2):
		type_list.append(2)
	for i in range (type3):
		type_list.append(3)

	count = 0
	while count <12000:
		random.shuffle(type_list)
		print (type_list[0:20])
		new_file = 'screw_' + str(count) + '.lmp'
		new_file = os.path.join(current_path, new_file)
		# copy_file(old_file, new_file)
		sub_atom_type(old_file, new_file, type_list)
		count+=1


def copy_file(original_file, new_file):
	shutil.copy(original_file, new_file)


def sub_atom_type(source_file, write_file, type_list):
	c = 0
	write_file = open(write_file, 'w')
	with open(source_file) as f:
		for i, line in enumerate(f):
			if i < 12:
				write_file.write(line)
			else:
				words = line.split()
				words[1] = str(type_list[c])
				write_file.write(' '.join(words))
				write_file.write('\n')
				c += 1


				
				

if __name__ == "__main__":
    # execute only if run as a script
    main()
