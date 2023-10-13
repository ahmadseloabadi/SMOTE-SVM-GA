import numpy as np
import pandas as pd
import random as rd
#import time
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn import svm
import Genetic_Algorithm as svm_hp_opt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import train_test_split

ulasan=pd.read_csv('data/dataset_bersih.csv')
X = ulasan['Stemming']
Y = ulasan['Sentimen']
vectorizer = TfidfVectorizer()
ulasan = vectorizer.fit_transform(X)
Encoder = LabelEncoder()
sentimen = Encoder.fit_transform(Y)

#penerapan smote
smote = SMOTE(sampling_strategy='auto',random_state=42)
X_smote, Y_smote = smote.fit_resample(ulasan, sentimen)

x=X_smote
y=Y_smote

# hyperparameters (user inputted parameters)
prob_crsvr = 0.6 # probablity of crossover
prob_mutation = 0.1 # probablity of mutation
population = 40 # population number
generations = 20 # generation number

kfold = 3

# x and y decision variables' encoding
# 12 genes for x and 12 genes for y (arbitrary number)
x_y_string = np.array([0,1,0,0,0,1,0,0,1,0,0,1,
                       0,1,1,1,0,0,1,0,1,1,1,0]) # initial solution


# create an empty array to put initial population
pool_of_solutions = np.empty((0,len(x_y_string)))


# create an empty array to store a solution from each generation
# for each generation, we want to save the best solution in that generation
# to compare with the convergence of the algorithm
best_of_a_generation = np.empty((0,len(x_y_string)+1))


# shuffle the elements in the initial solution (vector)
# shuffle n times, where n is the no. of the desired population
for i in range(population):
    rd.shuffle(x_y_string)
    pool_of_solutions = np.vstack((pool_of_solutions,x_y_string))


# so now, pool_of_solutions, has n (population) chromosomes


#start_time = time.time() # start time (timing purposes)

gen = 1 # we start at generation no.1 (tracking purposes)

for i in range(generations): # do it n (generation) times
    
    # an empty array for saving the new generation
    # at the beginning of each generation, the array should be empty
    # so that you put all the solutions created in a certain generation
    new_population = np.empty((0,len(x_y_string)))
    
    # an empty array for saving the new generation plus its obj func val
    new_population_with_obj_val = np.empty((0,len(x_y_string)+1))
    
    # an empty array for saving the best solution (chromosome)
    # for each generation
    sorted_best = np.empty((0,len(x_y_string)+1))
    
    print()
    print()
    print("--> Generation: #", gen) # tracking purposes
    
    
    family = 1 # we start at family no.1 (tracking purposes)
    
    
    for j in range(int(population/2)): # population/2 because each gives 2 parents
        
        print()
        print("--> Family: #", family) # tracking purposes
        
            
        # selecting 2 parents using tournament selection
        # "genf.find_parents_ts"[0] gives parent_1
        # "genf.find_parents_ts"[1] gives parent_2
        parent_1 = svm_hp_opt.find_parents_ts(pool_of_solutions,
                                              x=x,y=y)[0]
        parent_2 = svm_hp_opt.find_parents_ts(pool_of_solutions,
                                              x=x,y=y)[1]
        
        
        # crossover the 2 parents to get 2 children
        # "genf.crossover"[0] gives child_1
        # "genf.crossover"[1] gives child_2
        child_1 = svm_hp_opt.crossover(parent_1,parent_2,
                               prob_crsvr=prob_crsvr)[0]
        child_2 = svm_hp_opt.crossover(parent_1,parent_2,
                               prob_crsvr=prob_crsvr)[1]
        
        
        # mutating the 2 children to get 2 mutated children
        # "genf.mutation"[0] gives mutated_child_1
        # "genf.mutation"[1] gives mutated_child_2
        mutated_child_1 = svm_hp_opt.mutation(child_1,child_2,
                                      prob_mutation=prob_mutation)[0]
        mutated_child_2 = svm_hp_opt.mutation(child_1,child_2,
                                      prob_mutation=prob_mutation)[1]
        
        
        # getting the obj val (fitness value) for the 2 mutated children
        # "genf.fitness"[2] gives obj val for the mutated child
        obj_val_mutated_child_1 = svm_hp_opt.fitness(x=x,y=y,
                                                             chromosome=mutated_child_1,
                                                             kfold=kfold)[2]
        obj_val_mutated_child_2 = svm_hp_opt.fitness(x=x,y=y,
                                                             chromosome=mutated_child_2,
                                                             kfold=kfold)[2]
        
        
        # for each mutated child, put its obj val next to it
        mutant_1_with_obj_val = np.hstack((obj_val_mutated_child_1,
                                               mutated_child_1)) # lines 132 and 140
        
        mutant_2_with_obj_val = np.hstack((obj_val_mutated_child_2,
                                               mutated_child_2)) # lines 134 and 143
        
        
        # we need to create the new population for the next generation
        # so for each family, we get 2 solutions
        # we keep on adding them till we are done with all the families in one generation
        # by the end of each generation, we should have the same number as the initial population
        # so this keeps on growing and growing
        # when it's a new generation, this array empties and we start the stacking process
        # and so on
        # check line 88
        new_population = np.vstack((new_population,
                                    mutated_child_1,
                                    mutated_child_2))
        
        
        # same explanation as above, but we include the obj val for each solution as well
        # check line 91
        new_population_with_obj_val = np.vstack((new_population_with_obj_val,
                                                 mutant_1_with_obj_val,
                                                 mutant_2_with_obj_val))
        
        
        # after getting 2 mutated children (solutions), we get another 2, and so on
        # until we have the same number of the intended population
        # then we go to the next generation and start over
        # since we ended up with 2 solutions, we move on to the next possible solutions
        family = family+1
        
          
    # check line 60
    # check line 164
    # we replace the initial (before) population with the new one (current generation)
    # this new pool of solutions becomes the starting population of the next generation
    pool_of_solutions = new_population
    
    
    # for each generation
    # we want to find the best solution in that generation
    # so we sort them based on index [0], which is the obj val
    sorted_best = np.array(sorted(new_population_with_obj_val,
                                               key=lambda x:x[0]))
    
    
    # since we sorted them from best to worst
    # the best in that generation would be the first solution in the array
    # so index [0] of the "sorted_best" array
    best_of_a_generation = np.vstack((best_of_a_generation,
                                      sorted_best[0]))
    
    
    # increase the counter of generations (tracking purposes)
    gen = gen+1       



#end_time = time.time() # end time (timing purposes)



# check line 171
# for our very last generation, we have the last population
# for this array of last population (convergence), there is a best solution
# so we sort them from best to worst
sorted_last_population = np.array(sorted(new_population_with_obj_val,
                                         key=lambda x:x[0]))

sorted_best_of_a_generation = np.array(sorted(best_of_a_generation,
                                         key=lambda x:x[0]))

sorted_last_population[:,0] = 1-(sorted_last_population[:,0]) # get accuracy instead of error
sorted_best_of_a_generation[:,0] = 1-(sorted_best_of_a_generation[:,0])

# since we sorted them from best to worst
# the best would be the first solution in the array
# so index [0] of the "sorted_last_population" array
best_string_convergence = sorted_last_population[0]

best_string_overall = sorted_best_of_a_generation[0]

print()
#print()
#print("Execution Time in Minutes:",(end_time - start_time)/60) # exec. time


print()
print()
print("------------------------------")
print()
#print("Execution Time in Seconds:",end_time - start_time) # exec. time
#print()
print("Final Solution (Convergence):",best_string_convergence[1:]) # final solution entire chromosome
print("Encoded C (Convergence):",best_string_convergence[1:14]) # final solution x chromosome
print("Encoded Gamma (Convergence):",best_string_convergence[14:]) # final solution y chromosome
print()
print("Final Solution (Best):",best_string_overall[1:]) # final solution entire chromosome
print("Encoded C (Best):",best_string_overall[1:14]) # final solution x chromosome
print("Encoded Gamma (Best):",best_string_overall[14:]) # final solution y chromosome

# to decode the x and y chromosomes to their real values
final_solution_convergence = svm_hp_opt.fitness(x=x,y=y,
                                                        chromosome=best_string_convergence[1:],
                                                        kfold=kfold)

final_solution_overall = svm_hp_opt.fitness(x=x,y=y,
                                                    chromosome=best_string_overall[1:],
                                                    kfold=kfold)

# the "svm_hp_opt.fitness" function returns 3 things -->
# [0] is the x value
# [1] is the y value
# [2] is the obj val for the chromosome (avg. error)
print()
print("Decoded C (Convergence):",round(final_solution_convergence[0],5)) # real value of x
print("Decoded Gamma (Convergence):",round(final_solution_convergence[1],5)) # real value of y
print("Obj Value - Convergence:",round(1-(final_solution_convergence[2]),5)) # obj val of final chromosome
print()
print("Decoded C (Best):",round(final_solution_overall[0],5)) # real value of x
print("Decoded Gamma (Best):",round(final_solution_overall[1],5)) # real value of y
print("Obj Value - Best in Generations:",round(1-(final_solution_overall[2]),5)) # obj val of final chromosome
print()
print("------------------------------")

# test model ga svm
X_train, X_test, Y_train, Y_test = train_test_split(X_smote, Y_smote, test_size=0.20, random_state=42)

print("c :",final_solution_overall[0])
print("gamma :",final_solution_overall[1])
clf = svm.SVC(kernel='rbf',C=final_solution_overall[0], gamma=final_solution_overall[1])
clf.fit(X_train,Y_train)
test_accuracy = clf.score(X_test, Y_test)
print("Test accuracy:", test_accuracy)