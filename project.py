from pyeasyga import pyeasyga
import matplotlib.pyplot as plt
import numpy as np
import time
import copy

# global variables
BFS_wanted_chromosome = []
# setup data

# smile
# size = 2
# coreData = ([1, 1,
#              0, 1])

# size = 5
# coreData = ([1, 1, 0, 1, 1,
#             0, 0, 0, 0, 0,
#             0, 0, 1, 0, 0,
#             1, 0, 0, 0, 1,
#             0, 1, 1, 1, 0])
# size = 10
# coreData = ([0, 1, 0, 0, 1, 1, 1, 0, 1, 0,
#              1, 1, 0, 0, 1, 1, 1, 0, 1, 1,
#              0, 1, 0, 0, 0, 1, 0, 0, 1, 0,
#              0, 1, 1, 1, 1, 1, 1, 1, 1, 0,
#              0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
#              0, 0, 0, 0, 1, 1, 1, 1, 0, 0,
#              0, 0, 1, 1, 1, 1, 0, 1, 0, 0,
#              0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
#              0, 0, 1, 0, 0, 0, 0, 1, 1, 1,
#              1, 1, 1, 0, 0, 0, 0, 0, 0, 0])

# butterfly
#
size = 15
coreData = ([0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0,
            1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1,
            1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1,
            0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1,
            0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0,
            0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0,
            1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0,
            1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0,
            0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0])


def find_chromosome_withBFS():
    start_bfs = time.time()
    global BFS_wanted_chromosome
    for num in range(0, (pow(2, (size*size)) - 1)):
        if num % 50000000 == 0:
            print(str(num / pow(2, size * size) * 100) + " %")
        chromosome = bin(num)[2:].zfill(size*size)
        if check_if_chromosomes_are_same(list(chromosome)):
            BFS_wanted_chromosome = chromosome
            break
    end_bfs = time.time()
    print("BFS finding time: " + str(end_bfs-start_bfs))


def check_if_chromosomes_are_same(chromosome):
    for (org, tmp) in zip(coreData, chromosome):
        if org != int(tmp):
            return False
    return True


def printNonogram(array):
    result = ''
    for (index, item) in enumerate(array):
        if index % size == 0:
            result += "\n"
        if item:
            result += ' # '
        else:
            result += ' . '
    print(str(result))


printNonogram(coreData)  # print original image

# --------------------------------------------------------------------------------
# ga stuff based on pairs


def get_list_from_row(array):
    main_list = []
    for row in array:
        counter = 0
        h_list = []
        previous = 0
        for el in row:
            if el:
                counter += 1
            else:
                if previous:
                    h_list.append(counter)
                    counter = 0
            previous = el
        if previous:
            h_list.append(counter)
        if h_list:
            main_list.append(h_list)
        else:
            main_list.append([0])

    return main_list


def get_list_of_sums_from_array(array):
    main_list = []
    for row in array:
        tmp_sum = np.sum(row)
        main_list.append(tmp_sum)
    return main_list


def count_rows_and_columns(array):
    helper = np.array(array)
    arr_2d_row = np.reshape(helper, (size, size))
    arr_2d_col = np.reshape(helper, (size, size), order='F')

    return get_list_from_row(arr_2d_row), get_list_from_row(arr_2d_col)


counted_rows_columns = count_rows_and_columns(coreData)

print("Pairs in rows: " + str(counted_rows_columns[0]))
print("Pairs in cols: " + str(counted_rows_columns[1]))


def check_if_pairs_are_same(arr1, arr2):
    grade_h = 0
    for org, chrom in zip(arr1, arr2):
        if len(org) == len(chrom):
            flag = 1
            for (o, ch) in zip(org, chrom):
                if o != ch:
                    flag = 0
                    break
            if flag:
                grade_h += 1
    return grade_h


def fitness_based_on_pairs(individual, data):
    # 0-30 <- 30 is perfect chromosome
    grade = 0
    counted_r_c_chromosome = count_rows_and_columns(individual)

    grade += check_if_pairs_are_same(counted_rows_columns[0], counted_r_c_chromosome[0])
    grade += check_if_pairs_are_same(counted_rows_columns[1], counted_r_c_chromosome[1])

    return grade

# --------------------------------------------------------------------------------------------
# ga stuff based on sums


def check_if_sums_are_same(arr1, arr2):
    grade_h = 0
    for org, chrom in zip(arr1, arr2):
        if org == chrom:
            grade_h += 1
    return grade_h


def fitness_based_on_sums(individual, data):
    # 0-30 <- 30 is perfect chromosome
    grade = 0
    summed_r_c_chromosome = sum_rows_and_columns(individual)

    grade += check_if_sums_are_same(counted_sums_of_rows_and_cols[0], summed_r_c_chromosome[0])
    grade += check_if_sums_are_same(counted_sums_of_rows_and_cols[1], summed_r_c_chromosome[1])

    return grade


def sum_rows_and_columns(array):
    helper = np.array(array)
    arr_2d_row = np.reshape(helper, (size, size))
    arr_2d_col = np.reshape(helper, (size, size), order='F')

    return get_list_of_sums_from_array(arr_2d_row),  get_list_of_sums_from_array(arr_2d_col)


counted_sums_of_rows_and_cols = sum_rows_and_columns(coreData)
print("Sums in rows: " + str(counted_sums_of_rows_and_cols[0]))
print("Sums in cols: " + str(counted_sums_of_rows_and_cols[1]))

# --------------------------------------------------------------------------------------------


ga = pyeasyga.GeneticAlgorithm(coreData,
                               generations=500,
                               mutation_probability=0.8,
                               crossover_probability=0.8,
                               population_size=500,
                               elitism=True
                                )


def run(self):
    """Run (solve) the Genetic Algorithm."""
    count_avg = lambda arr: sum(el.fitness for el in arr) / len(arr)

    self.data_collected_while_running = []
    self.create_first_generation()
    for _ in range(1, self.generations):
        self.create_next_generation()
        best_fitness_chromosome = self.current_generation[0].fitness
        average_of_chromosomes = count_avg(self.current_generation)
        self.data_collected_while_running.append((best_fitness_chromosome, average_of_chromosomes))


setattr(pyeasyga.GeneticAlgorithm, "run", run)

ga.fitness_function = fitness_based_on_pairs              # set the GA's fitness function
# ga.fitness_function = fitness_based_on_sums              # set the GA's fitness function

start = time.time()
ga.run()                                    # run the GA
end = time.time()
time_of_execution = end - start
print("Time of execution = " + str(time_of_execution))
print(ga.best_individual())
printNonogram(ga.best_individual()[1])
print(count_rows_and_columns(ga.best_individual()[1]))


# find_chromosome_withBFS()
# print("\n\n\nWanted chromosome found in BFS = " + str(BFS_wanted_chromosome))


best_fitness = [el[0] for el in ga.data_collected_while_running]
avg_fitness = [el[1] for el in ga.data_collected_while_running]
x_axis = range(1, ga.generations)

plt.plot(x_axis, best_fitness, label='BEST', color='red')
plt.xlabel('Generations  || size: ' + str(size))
plt.ylabel('Grade of fitness function')

plt.plot(x_axis, avg_fitness, label='AVG', color='blue')
plt.legend()
plt.show()
