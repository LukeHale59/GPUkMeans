import os
# define input and output file names
input_file = 'wine.txt'
output_file = 'output/output.txt'

if not os.path.exists('output'):
    os.mkdir('output')

# define the number of points to include in each row
#num_points = 5

for num_points in range(1,14):
    # read input file and process data
    with open(input_file, 'r') as f_in, open('output/wine_'+str(num_points)+'.txt', 'w') as f_out:
        # write first line to output file
        first_line = f_in.readline()
        f_out.write(first_line)

        # process remaining lines
        for line in f_in:
            # split line into fields and get first `num_points` fields
            fields = line.strip().split()
            first_fields = fields[:num_points]

            # write first `num_points` fields to output file
            f_out.write('\t'.join(first_fields) + '\n')

