import numpy as np
from sklearn.model_selection import KFold ,cross_val_score
from sklearn import svm



# menghitung nilai fitness untuk kromosom 0s dan 1s
def fitness(x,y,chromosome,kfold=5):  
    
    # x = c
    lb_x = 1 # batas bawah kromosom x
    ub_x = 50 # batas atas kromosom x
    len_x = (len(chromosome)//2) # panjang kromosom x
    
    # y = gamma
    lb_y = 0.1 # batas bawah kromosom y
    ub_y = 0.99 # batas atas kromosom y
    len_y = (len(chromosome)//2) # panjang kromosom y
    
    precision_x = (ub_x-lb_x)/((2**len_x)-1) # presisi untuk decoding x
    precision_y = (ub_y-lb_y)/((2**len_y)-1) # presisi untuk decoding y
    
    z = 0 # karena kita mulai dari 2^0, dalam rumus
    t = 1 # karena kita mulai dari elemen terakhir vektor [indeks -1]
    x_bit_sum = 0 # inisiasi (sum(bit)*2^i adalah 0 pada awalnya)
    for i in range(len(chromosome)//2):
        x_bit = chromosome[-t]*(2**z)
        x_bit_sum = x_bit_sum + x_bit
        t = t+1
        z = z+1   
    
    z = 0 # karena kita mulai dari 2^0, dalam rumus
    t = 1 + (len(chromosome)//2) # [6,8,3,9] (2 pertama adalah y, jadi indeksnya adalah 1+2 = -3)
    y_bit_sum = 0 # inisiasi (sum(bit)*2^i adalah 0 pada awalnya)
    for j in range(len(chromosome)//2):
        y_bit = chromosome[-t]*(2**z)
        y_bit_sum = y_bit_sum + y_bit
        t = t+1
        z = z+1
    
    # rumus untuk memecahkan kode kromosom 0s dan 1s menjadi bilangan aktual, nilai x atau y
    c_hyperparameter = (x_bit_sum*precision_x)+lb_x
    gamma_hyperparameter = (y_bit_sum*precision_y)+lb_y
       
    kf = KFold(n_splits=kfold)
    
    # objective function value for the decoded x and decoded y
    sum_of_error = 0
    for train_index,test_index in kf.split(x,y):
        x_train,x_test = x[train_index],x[test_index]
        y_train,y_test = y[train_index],y[test_index]
        
        model = svm.SVC(kernel="rbf",
                        C=c_hyperparameter,
                        gamma=gamma_hyperparameter)
        model.fit(x_train,np.ravel(y_train))
        
        accuracy = model.score(x_test,y_test)
        error = 1-(accuracy)
        sum_of_error += error
        
    avg_error = sum_of_error/kfold
    
    # fungsi yang ditentukan akan mengembalikan 3 nilai
    return c_hyperparameter,gamma_hyperparameter,avg_error




# menemukan 2 parent dari kumpulan solusi
# menggunakan metode tournament selection  
def find_parents_ts(all_solutions,x,y):
    
    # membuat array kosong untuk parent yang di pilih
    parents = np.empty((0,np.size(all_solutions,1)))
    
    for i in range(2): # melakukan proses untuk mendapatkan 2 parent
        
        # memilih 3 parent acak dari kumpulan solusi yang Anda miliki
        
        # mendapatkan 3 integer
        indices_list = np.random.choice(len(all_solutions),3,
                                        replace=False)
        
        # dapatkan 3 kemungkinan parent untuk selection
        posb_parent_1 = all_solutions[indices_list[0]]
        posb_parent_2 = all_solutions[indices_list[1]]
        posb_parent_3 = all_solutions[indices_list[2]]
        
        
        # dapatkan nilai fitness untuk setiap kemungkinan parent        
        # index no.2 karena fungsi fitness memberikan nilai fitness pada index no.2
        obj_func_parent_1 = fitness(x=x,y=y,chromosome=posb_parent_1)[2] # kemungkianan parent 1
        obj_func_parent_2 = fitness(x=x,y=y,chromosome=posb_parent_2)[2] # kemungkianan parent 2
        obj_func_parent_3 = fitness(x=x,y=y,chromosome=posb_parent_3)[2] # kemungkianan parent 3
        
        
        # mencari parent mana yang terbaik
        min_obj_func = min(obj_func_parent_1,obj_func_parent_2,
                           obj_func_parent_3)
        
        if min_obj_func == obj_func_parent_1:
            selected_parent = posb_parent_1
        elif min_obj_func == obj_func_parent_2:
            selected_parent = posb_parent_2
        else:
            selected_parent = posb_parent_3
        
        # letakkan parent yang dipilih dalam array kosong yang kami buat di atas
        parents = np.vstack((parents,selected_parent))
        
    parent_1 = parents[0,:] # parent_1, elemen pertama di dalam array
    parent_2 = parents[1,:] # parent_2, elemen kedua di dalam array
    
    return parent_1,parent_2 # fungsi akan mengembalikan 2 nilai array



# crossover antara  2 parents untuk membuat 2 children
# input functions adalah parent_1, parent_2, dan  probability dari crossover
# default probability dari crossover adalah 1
def crossover(parent_1,parent_2,prob_crsvr):
    
    child_1 = np.empty((0,len(parent_1)))
    child_2 = np.empty((0,len(parent_2)))
    
    
    rand_num_to_crsvr_or_not = np.random.rand() # apakah kita melakukan crossover atau tidak???
    
    if rand_num_to_crsvr_or_not < prob_crsvr:
        index_1 = np.random.randint(0,len(parent_1))
        index_2 = np.random.randint(0,len(parent_1))
        
        # mendapatkan indices yang berbeda untuk memastikan Anda menyilangkan setidaknya satu gen
        
        while index_1 == index_2:
            index_2 = np.random.randint(0,len(parent_1))
        
        index_parent_1 = min(index_1,index_2) 
        index_parent_2 = max(index_1,index_2) 
            
            
        ### untuk PARENT_1 ###
        
        # first_seg_parent_1 -->
        # untuk parent_1: genes yang bermula dari awal parent_1 ke pertengahan segmen parent_1
        
        first_seg_parent_1 = parent_1[:index_parent_1]
        
        # middle segment; letak crossover terjadi
        # for parent_1: genes dari index yang di pilih untuk parent_1 ke index yang di pilih dari parent_1
                
        mid_seg_parent_1 = parent_1[index_parent_1:index_parent_2+1]
        
        # last_seg_parent_1 -->
        # for parent_1: genes dari ujung segmen tengah dari parent_1 ke akhir genes dari parent_1
  
        last_seg_parent_1 = parent_1[index_parent_2+1:]
        
        
        ### FOR PARENT_2 ###
        
        # first_seg_parent_2 --> sama seperti parent_1
        first_seg_parent_2 = parent_2[:index_parent_1]
        
        # mid_seg_parent_2 --> sama seperti parent_1
        mid_seg_parent_2 = parent_2[index_parent_1:index_parent_2+1]
        
        # last_seg_parent_2 --> sama seperti parent_1
        last_seg_parent_2 = parent_2[index_parent_2+1:]
        
        
        ### membuat CHILD_1 ###
        
        # segmen pertama dari parent_1
        # ditambah segmen tengah dari parent_2
        # di tambah segmen akhir dari parent_1
        child_1 = np.concatenate((first_seg_parent_1,mid_seg_parent_2,
                                  last_seg_parent_1))
        
        
        ### membuat CHILD_2 ###
        
        # segmen pertama dari parent_2
        # ditambah segmen tengah dari parent_1
        # di tambah segmen akhir dari parent_2
        child_2 = np.concatenate((first_seg_parent_2,mid_seg_parent_1,
                                  last_seg_parent_2))
    
    
    # syarat ketika kita tidak mau melakukan crossover
    # ketika rand_num_to_crsvr_or_not tidak kurang atau lebih besar dari prob_crsvr
    # akan tetapi ketika prob_crsvr == 1, lalu rand_num_to_crsvr_or_not akan selalu kurang dari prob_crsvr, jadi akan selalu di lakukan crossover
            
    else:
        child_1 = parent_1
        child_2 = parent_2
    
    return child_1,child_2 # fungsi akan mengembalikan 2 array



############################################################
### MUTATING 2 CHILDREN untuk membuat MUTATED CHILDREN ###
############################################################

# mutation dari 2 children
# input functions adalah child_1, child_2, dan probability dari mutation
# default probability dari mutation is 0.2
def mutation(child_1,child_2,prob_mutation):
    
    # mutated_child_1
    mutated_child_1 = np.empty((0,len(child_1)))
      
    t = 0 # mulai dari indeks paling pertama child_1
    for i in child_1: # untuk setiap gene (index)
        
        rand_num_to_mutate_or_not = np.random.rand() # apakah kita bermutasi atau tidak???
        
        # jika rand_num_to_mutate_or_not kurang dari kemungkinan mutasi 
        # lalu kita bermutasi pada gen yang diberikan itu (indeks tempat kita berada saat ini)
        if rand_num_to_mutate_or_not < prob_mutation:
            
            if child_1[t] == 0: # jika kita mutate, a 0 menjadi a 1
                child_1[t] = 1
            
            else:
                child_1[t] = 0  # jika kita mutate, a 1 menjadi a 0
            
            mutated_child_1 = child_1
            
            t = t+1
        
        else:
            mutated_child_1 = child_1
            
            t = t+1
    
       
    # mutated_child_2
    # proses yang sama pada mutated_child_1
    mutated_child_2 = np.empty((0,len(child_2)))
    
    t = 0
    for i in child_2:
        
        rand_num_to_mutate_or_not = np.random.rand() # probability dari mutate
        
        if rand_num_to_mutate_or_not < prob_mutation:
            
            if child_2[t] == 0:
                child_2[t] = 1
           
            else:
                child_2[t] = 0
            
            mutated_child_2 = child_2
            
            t = t+1
        
        else:
            mutated_child_2 = child_2
            
            t = t+1
    
    return mutated_child_1,mutated_child_2 # fungsi akan mengembalikan 2 arrays