import numpy as np
import matplotlib.pyplot as plt
import itertools
import sys
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def main():
    if len(sys.argv) < 3:
        print("Need at least a system function, number of variables and number of time steps")
        return
    system = globals()[sys.argv[1]]
    vars = sys.argv[2]
    t = int(sys.argv[3])
    basic_operations = {1: vars, 3: [add, multiply]}
    try:
        x = np.array([[float(d) for d in a[1:-1].split(',')] for a in sys.argv[4:]]) if len(sys.argv) > 4 else np.random.rand(10,len(vars))-0.5 # each row is one initial_point
        input_data, output_data = generate_data(x, system, t)
        input_train_data, input_test_data, output_train_data, output_test_data = separate_train_test_data(input_data, output_data)
    except ValueError:
        print('Given points are not of good shape')
        return
    res_program, res_coeff, res_score = get_equations(input_train_data, output_train_data, input_test_data, output_test_data, basic_operations, LinearRegression())
    print('--------------------\n*** RESULT BELOW ***\n--------------------')
    print('kernel =',res_program[0],'of size',res_program[1],'\ncoeff (first column is constant)=', res_coeff, '\ntest_score =',res_score)

def dot_system1(x):
    return np.array([x[0] - 0.1*x[0]*x[1],
                     -1.5*x[1] + 0.075*x[0]*x[1]])

def dot_system2(x):
    return np.array([1 - x[0] - 0.25*x[0]*x[1],
                     -x[1] + 2*x[1]*x[2],
                     0.25*x[0] - 2*x[1]**3])

def generate_data(x, system, t):
    ''' Generate t more data points from multiple initial points, x of shape=(nb_points,nb_vars)
        Return data of shape=(nb_points,t+1,nb_vars)
    '''
    if x.ndim < 2:
        x = np.reshape(x,(1,-1))
    input_data = None
    output_data = None
    for i in range(x.shape[0]): # for each initial point
        data = np.concat((x[i:i+1],np.zeros((t, x.shape[1]))))
        for j in range(1,t+1):
            data[j] += system(data[j-1]) + data[j-1]
        data = np.reshape(data,(1,*data.shape))
        input_data = data[:,:-1] if input_data is None else np.concat((input_data,data[:,:-1]))
        output_data = data[:,1:] if output_data is None else np.concat((output_data,data[:,1:]))
    return input_data, output_data

def separate_train_test_data(input_data, output_data, test_size=0.2):
    ''' Return data of shape=(nb_data,nb_vars)
    '''
    input_data = np.reshape(input_data,(-1,input_data.shape[2]))
    output_data = np.reshape(output_data,(-1,output_data.shape[2]))
    return train_test_split(input_data, output_data, test_size=test_size)

def show_data(data, var_names='xyz', print_data=False):
    if print_data:
        print('data',data)
    fig, ax = plt.subplots(1, len(data), figsize=(8,2))
    for i in range(data.shape[0]):
        for j in range(data.shape[2]):
            ax[i].plot(data[i,:,j], label = var_names[j], linewidth = 1)
            ax[i].scatter(np.arange(data.shape[1]), data[i,:,j], marker='x')
        plt.xlabel('time')
        ax[i].legend()
    plt.show()

def add(x,y):
    ''' Addition to set (to ignore repeat variable (coefficients will be learned later))
        Returns set(represent sum) of products or variables
        x, y : sets
    '''
    # print(x,y)
    if (x == y) or x.issubset(y) or y.issubset(x):
        return False
    # print('x =',x,'y =',y,'\t--> x+y =',x|y)
    return x|y

def multiply(x,y):
    ''' Broadcast multiplication
        Return set(represent sum) of products or variables
        x, y: sets
    '''
    res = list(itertools.product(x,y))
    res = {'*'.join(sorted(p)) for p in res}
    max_degree = max({len(p.split('*')) for p in res})
    # print('x =',x,'y =',y,'\t--> x*y =',res)
    return res if max_degree <= 3 else False

def combine_all(program_bank, size, latest_round):
    latest_programs = [(pg,s,r) for pg,s,r in program_bank if r==latest_round]
    old_programs = [(pg,s,r) for pg,s,r in program_bank if r<latest_round]
    # avoid redo old combinations
    latest_combinations = list(itertools.combinations_with_replacement(latest_programs, size-1))
    latest_products = [(l,*o) for l,o in itertools.product(latest_programs, itertools.combinations_with_replacement(old_programs,size-2))]
    return latest_combinations + latest_products

def combine_latest_only(program_bank, size, latest_round):
    latest_programs = [(pg,s,r) for pg,s,r in program_bank if r==latest_round]
    latest_combinations = list(itertools.combinations_with_replacement(latest_programs, size-1))
    return latest_combinations

def combine_programs(program_bank, basic_operations, strategy):
    ''' Returns new programs '''
    new_programs = []
    latest_round = max([r for _,_,r in program_bank])
    for size in basic_operations:
        if size > 1:
            combinations = strategy(program_bank, size, latest_round)
            
            # perform operations on found combinations
            for op in basic_operations[size]:
                combined_programs = [(op(x[0],y[0]),x[1]+y[1]+1,latest_round+1) for x,y in combinations]
                combined_programs = [(pg,s,r) for pg,s,r in combined_programs if pg]
                new_programs += combined_programs
    return remove_redundant(program_bank, new_programs)

def check_sub_program(program_bank, pg):
    for program,_,_ in program_bank:
        if pg == program or pg.issubset(program):
            return True
    return False
        
def remove_redundant(program_bank, new_programs):
    ''' Check observational equivalence and remove redundant programs
    '''
    new_programs = [(pg,s,r) for pg,s,r in new_programs if not check_sub_program(program_bank, pg)]
    #TODO check redundant using set of tuple(sorted list)???
    non_redudant_programs = [] # list of sets
    non_redundant_index = [] # index in new programs
    for i in range(len(new_programs)):
        if new_programs[i][0] in non_redudant_programs:
            existing_index = non_redudant_programs.index(new_programs[i][0])
            if new_programs[i][1] < new_programs[existing_index][1]: # keep if smaller size program
                non_redundant_index[existing_index] = i
        else:
            non_redudant_programs.append(new_programs[i][0])
            non_redundant_index.append(i)
    new_programs = [new_programs[i] for i in range(len(new_programs)) if i in non_redundant_index]
    return new_programs

def program_bank_dict_by_size(program_bank):
    program_bank_dict = dict()
    for pg,s,_ in program_bank:
        if s not in program_bank_dict:
            program_bank_dict[s] = [pg]
        else:
            program_bank_dict[s] += [pg]
    return program_bank_dict

def kernel_transform(input, program, var_names='xyz'):
    input_kernel = np.ones((input.shape[0],1))
    for pg in program:
        if pg in var_names: # if variable
            new_input = input[:,var_names.index(pg)]
        else: # if product
            vars_to_multiply = [input[:,var_names.index(var)] for var in pg.split('*')]
            new_input = np.prod(vars_to_multiply, axis=0)
        new_input = np.reshape(new_input,(input.shape[0],1))
        input_kernel = np.concat((input_kernel,new_input), axis=1)
    return input_kernel

def get_equations(input_train, output_train, input_test, output_test, basic_operations, regression_model=LinearRegression(),  combine_strategy=combine_all, expected_test_score=0.99, var_names='xyz', verbose=False):
    program_bank = [({op},1,1) for op in basic_operations[1]]
    new_programs = program_bank[:]
    res_score = 0
    res_coeff = None
    res_program = None
    while res_score < expected_test_score:
        for program in new_programs:
            program = (sorted(program[0]),program[1])
            print('regression with kernel',program, end='\t')

            # transform input according to synthesized program
            input_train_kernel = kernel_transform(np.copy(input_train), program[0], var_names)
            input_test_kernel = kernel_transform(np.copy(input_test), program[0], var_names)

            # fit the model
            reg = regression_model.fit(input_train_kernel,output_train)
            print('train_score =',reg.score(input_train_kernel,output_train), end='\t')

            # test the model
            new_score = reg.score(input_test_kernel,output_test)
            print('test_score =', new_score)
            # if verbose:
                # print('params',reg.coef_)

            if new_score > res_score:
                res_score = new_score 
                res_program = program
                res_coeff = np.where(np.abs(reg.coef_) < 10**-10 , 0, reg.coef_)
            if res_score >= expected_test_score:
                break

        # continue finding new program by combining existing programs
        if res_score < expected_test_score:
            new_programs = combine_programs(program_bank, basic_operations, combine_strategy)
            program_bank += new_programs

    return res_program, res_coeff, res_score

if __name__ == "__main__":
    main()
