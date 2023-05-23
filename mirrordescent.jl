
# Caluclate the return and runtime of the Mirror descent algorithm
#UAI2023


using Pkg
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("DataFramesMeta")

using CSV
using DataFrames, DataFramesMeta
using CSV: File

# get discount factor
function get_discount(filename)
    frame = DataFrame(File(filename)); #t_df: dataFrame of training.csv
    return frame[1,2]
end

# Get state_space, action_space and model_space
function get_state_action_model_space(frame)

    state_space = [] # store all states, states are integer
    action_space = [] # store all actions, actions are integer
    model_space = [] #store all models, models are integer

    for i in 1:size(frame,1)

        if !(frame.idstateto[i] in state_space)
                push!(state_space,frame.idstateto[i] )
        end

        if !(frame.idstatefrom[i] in state_space)
                push!(state_space,frame.idstatefrom[i] )
        end
        if !(frame.idaction[i] in action_space)
                push!(action_space,frame.idaction[i] )
        end
        if !(frame.idoutcome[i]  in model_space)
                push!(model_space,frame.idoutcome[i] )
        end
            
    # sorted three lists in ascending order. Then index of a list has the same
    # value with the value in that index.For example, state_space[2] = 2
    state_space = sort(state_space)
    action_space = sort(action_space)
    model_space = sort(model_space)
  end
    return state_space,action_space, model_space
end

# get initial distribution of states
function get_inital_state_distribution(frame1, state_space)
    statecount = length(state_space)
    
    # key: state ; value: probability
    ini_states =  Dict()
    
    for i in 1:size(frame1,1)
       ini_states[frame1.idstate[i]] = frame1.probability[i]
    end

    for j in 1:statecount
        if !(j in keys(ini_states))
                ini_states[j] = 0   
        end
    end
    return ini_states
end

function calculate_reward_probability(frame,state_space, action_space,model_space )
        
    state_size = length(state_space)
    action_size = length(action_space)
    model_size = length(model_space)
    
    # reward of going from state s to state s’ through action a.
    r = zeros((state_size, action_size,state_size,model_size))
    # transition probablity of (state, action, next state, model)
    p = zeros((state_size, action_size,state_size,model_size))

    
    for i in 1:size(frame,1)
        state_from = frame.idstatefrom[i]
        action = frame.idaction[i]
        state_to =  frame.idstateto[i]
        model =  frame.idoutcome[i]
        reward =  frame.reward[i]
        probability =  frame.probability[i]

        p[state_from, action, state_to, model] = frame.probability[i]
        r[state_from, action, state_to,model] = frame.reward[i]

    end      
    # normalize transition probabilities
    p_n = zeros((state_size,action_size,state_size, model_size))
    for m in model_space
        for s_from in state_space
            for a in action_space
                sum_t = sum([p[s_from,a,s_next, m] for s_next in state_space])
                if sum_t > 1e-15
                    for s_next in state_space
                        p_n[s_from,a,s_next,m] = p[s_from,a,s_next,m]/sum_t  
                    end
                end
            end
        end
  end
    return  r,p_n
end


#calculate weights of states over all models
function calculate_model_lamda(T,p,model_space,action_space,state_space,initial_policy,ini_states)
    
    model_size = length(model_space)
    state_size = length(state_space)
    
    model_lamda = zeros(Float64,(T+1,model_size,state_size))
    for m in model_space
        for s in state_space
                    model_lamda[1,m,s] = 1 / model_size * ini_states[s]
        end
    end
                
    for t in 2:T       
        for snext in state_space
            for m in model_space
                    for s in state_space
                        for a in action_space
                            if initial_policy[t-1,s,a] > 1e-15
                                model_lamda[t,m,snext] += initial_policy[t-1,s,a]* 
                                (p[s,a,snext,m] * model_lamda[t-1,m,s])
                            end
                        end
                    end
                
            end
        end
    end

    return model_lamda
end



#Initial policy: actions are uniformally taken in a state s
function ini_extract_policy(T, state_space,action_space, 
                     model_space, r, p, discount)
    
    model_size = length(model_space)
    state_size = length(state_space)
    action_size = length(action_space)
    
    pro= 1/ action_size # All actions are uniformally taken in a state s
    policy = zeros(Float64, (T+1, state_size,action_size))
    
    for t in 1:T
            for s in state_space
                for a in action_space
                    policy[t,s,a] = pro
                end
            end
    end
                    
    q_value= zeros(Float64,(T+1, model_size,state_size,action_size))
    update_state_Value = zeros(Float64,(length(model_space), length(state_space),T+2))
    t = T # time step
    while t>= 1
        for m in model_space
            for s in state_space
                for a in action_space
                    if policy[t,s,a] > 1e-15 # randomized policy
                        #Calculate further rewards
                        future_reward = policy[t,s,a] * sum([p[s,a,s_next,m] * 
                                          update_state_Value[m,s_next, t+1]
                                          for s_next in state_space])
                        # Calculate immediate rewards   
                        immediate_reward = policy[t,s,a] * sum([r[s,a,s_next,m] * p[s,a,s_next,m]
                                           for s_next in state_space])  
                        # Sum rewards of all available actions in state s
                        update_state_Value[m,s,t] += immediate_reward + discount  * future_reward  
                        q_value[t,m,s,a] = immediate_reward + discount  * future_reward
                    end
                end
            end
        end
        t=t-1 
    end
                     
    return policy, q_value
end
#policy at next iteration
function policy_next_iteration(initial_policy,q_value,ml,model_space,action_space,state_space,T,alpha,
                          r, p, discount)
    
    state_size = length(state_space)
    action_size = length(action_space)
    model_size = length(model_space)
    
    policy = zeros(Float64,(T+1, state_size,action_size))
    sat = zeros(Float64,(T+1, state_size,action_size))
    st = zeros(Float64,(T+1, state_size))
    sum_b_q = zeros(Float64,(T+1, state_size,action_size))
    max_values = zeros(Float64,(T+1, state_size))

    for t in 1:T
        for s in state_space
            max_value = -Inf64
            for a in action_space
                #s_m = 0.0
                for m in model_space
                    sum_b_q[t,s,a] += ml[t,m,s] * q_value[t,m,s,a]
                end
                if sum_b_q[t,s,a] > max_value
                    max_value = sum_b_q[t,s,a]
                end
            end
        max_values[t,s] = max_value
        end
    end
    
    for t in 1:T
        for s in state_space
            for a in action_space           
                sat[t,s,a] =initial_policy[t,s,a] * exp(alpha * (sum_b_q[t,s,a] -max_values[t,s] ))
                st[t,s] += sat[t,s,a]
            end
        end
    end
                
    for t in 1:T
        for s in state_space
            for a in action_space
                    policy[t,s,a] =  sat[t,s,a] /st[t,s]
            end
        end
    end
    
    
    q_value= zeros(Float64,(T+1, model_size,state_size,action_size))
    update_state_Value = zeros(Float64,(length(model_space), length(state_space),T+2))
    t = T # time step
    while t>= 1
        for m in model_space
            for s in state_space
                for a in action_space
                    if policy[t,s,a] > 1e-15 # randomized policy
                        #Calculate further rewards
                        future_reward = policy[t,s,a] * sum([p[s,a,s_next,m] * 
                                          update_state_Value[m,s_next, t+1]
                                          for s_next in state_space])
                        # Calculate immediate rewards   
                        immediate_reward = policy[t,s,a] * sum([r[s,a,s_next,m] * p[s,a,s_next,m]
                                           for s_next in state_space])  
                        # Sum rewards of all available actions in state s
                        update_state_Value[m,s,t] += immediate_reward + discount  * future_reward  
                        q_value[t,m,s,a] = immediate_reward + discount  * future_reward
                    end
                end
            end
        end
                                          
        t=t-1 
    end
        
    return policy, q_value
end

function  kl_value(initial_policy,updated_policy,model_space,
                                      action_space,state_space,T)
  
    sum_kl = 0.0
    for t in 1:T
        for s in state_space
            for a in action_space
                temp = 0
                if updated_policy[t,s,a] > 1e-10 && initial_policy[t,s,a] > 1e-10
                    temp = log((updated_policy[t,s,a]/initial_policy[t,s,a]))
                end
                if  updated_policy[t,s,a] < 1e-10
                    temp = 0.0
                end
                sum_kl += updated_policy[t,s,a] * temp
            end
        end
    end
    
    
    return sum_kl
end


# Test the policy on testing data 
# policy: the policy generated on training data; r: rewards;p: transition probability
# update_state_value: state values of all models. This table is overwritten 
# at every step. The final result of this table is the state values of all models
# at step 1. This algorithm starts at step 1, instead of 0
function evaluate_policy(T,policy,r,p,state_space, action_space, model_space, 
                     update_state_value, discount)
    
    t = T
    while t>= 1
        for m in model_space
            for s in state_space
                for a in action_space
                    if policy[t,s,a] > 1e-15 # randomized policy
                        #Calculate further reward
                        future_reward = policy[t,s,a] * sum([p[s,a,s_next,m] * update_state_value[m,s_next,t+1]
                                         for s_next in state_space ])
                        #Calculate the immediate reward   
                        immediate_reward = policy[t,s,a] * sum([r[s,a,s_next,m] * p[s,a,s_next,m]
                                         for s_next in state_space])
                        # update the state value of s in model m at step t                
                        update_state_value[m,s,t] += immediate_reward + discount * future_reward
                    end
                end
            end
        end
                        
        t=t-1
    end
            
    v = zeros(Float64,(length(model_space), length(state_space)))
    for m in model_space
        for s in state_space
            v[m,s] = update_state_value[m,s,1] 
        end
    end
        
    return v
end

function calculate_total_reward(values_of_states,ini_states,model_space,state_space)
    
    len_state = length(state_space)
    len_model = length(model_space)
    state_value_mean = zeros(len_state)
    total_reward = 0

    return_models =zeros( length(model_space))
    model_num = []
    #calculate standard deviation
    for m in 1:len_model
        push!(model_num,m)
        for s in 1:len_state
            return_models[m] += values_of_states[m,s] * ini_states[s]
        end
    end
    # write results to file
    all_mp = joinpath(@__DIR__,"resultfiles","all_returns_Mirror_descent_.csv")
    all_ab = DataFrame(Models=model_num,Return=return_models)
    CSV.write(all_mp, all_ab)
    
    for s in 1:len_state # 20: 0-19
        temp = 0.0 
        for r in 1:len_model
                temp = temp + values_of_states[r,s] 
        end
        #Given a state s, the average of s state values of all models
        state_value_mean[s] = temp/len_model
    end  
    # calcualte total rewards of a policy  
    total_reward = sum([state_value_mean[i] * ini_states[i]
                           for i in 1:len_state])
    
    return total_reward
end

  

function main()
    
    #domains = ['r','s','p','h','i']
    domains = ['r']
    for domain in domains
        #TS =[50,75,100,150]
        TS = [50]
        for T in TS
        
        runtime= @elapsed begin #start of the runtime
        
        initial_T = T


        # read discount factor from the file
        if(domain == 'r')
                discount_file = joinpath(@__DIR__,"domain","riverswim","parameters.csv")
        end
        if (domain == 's')
                discount_file = joinpath(@__DIR__,"domain","population_small","parameters.csv")
        end
        if (domain == 'p')
                discount_file = joinpath(@__DIR__,"domain","population","parameters.csv")
        end
        if (domain == 'h')
                discount_file = joinpath(@__DIR__,"domain","hiv","parameters.csv")
        end
        if (domain == 'i')
            discount_file = joinpath(@__DIR__,"domain","inventory","parameters.csv")
        end
                # Get the discount value
        discount = get_discount(discount_file)
        
         #Read training data from the file
        if(domain == 'r')
                train_file = joinpath(@__DIR__,"domain","riverswim", "training.csv");
        end
       if(domain == 's')
               train_file = joinpath(@__DIR__,"domain","population_small", "training.csv");
       end
       if(domain == 'p')
        train_file = joinpath(@__DIR__,"domain","population", "training.csv");
       end
       if(domain == 'h')
               train_file = joinpath(@__DIR__,"domain","hiv", "training.csv");
       end
       if(domain == 'i')
        train_file = joinpath(@__DIR__,"domain","inventory", "training.csv");
       end
        # Read a training file and offset relevant indices by one
        train_df = DataFrame(File(train_file)); #t_df: dataFrame of training.csv
        train = @transform(train_df, :idstatefrom = :idstatefrom .+1, :idaction = :idaction .+ 1,
                       :idstateto = :idstateto .+1, :idoutcome = :idoutcome .+ 1);
        state_space, action_space,model_space = get_state_action_model_space(train)

        # calculate rewards r, and transition probability trans_p
        r,trans_p = calculate_reward_probability(train,state_space,
                                             action_space,model_space)
        #Read initial distribution of states from a file
        if(domain == 'r')
                ini_file = joinpath(@__DIR__,"domain","riverswim","initial.csv")
        end
        if(domain == 's')
                ini_file = joinpath(@__DIR__,"domain","population_small","initial.csv")
        end
        if(domain == 'p')
                ini_file = joinpath(@__DIR__,"domain","population","initial.csv")
        end
        if(domain == 'h')
                ini_file = joinpath(@__DIR__,"domain","hiv","initial.csv")
        end
        if(domain == 'i')
            ini_file = joinpath(@__DIR__,"domain","inventory","initial.csv")
        end
        # Get the initial distribution of states
        ini_df = DataFrame(File(ini_file)); #t_df: dataFrame of training.csv
        initial = @transform(ini_df, :idstate = :idstate .+1);
        ini_states = get_inital_state_distribution(initial, state_space)  
        
        #Read data from the test file
        if(domain == 'r')
                test_file = joinpath(@__DIR__,"domain","riverswim","test.csv")
        end
        if(domain == 's')
                test_file = joinpath(@__DIR__,"domain","population_small","test.csv")
        end
        if(domain == 'p')
                test_file = joinpath(@__DIR__,"domain","population","test.csv")
        end
        if(domain == 'h')
                test_file = joinpath(@__DIR__,"domain","hiv","test.csv")
        end
        if(domain == 'i')
            test_file = joinpath(@__DIR__,"domain","inventory","test.csv")
        end
        test_df = DataFrame(File(test_file)); #t_df: dataFrame of training.csv
        test = @transform(test_df, :idstatefrom = :idstatefrom .+1, :idaction = :idaction .+ 1,
                       :idstateto = :idstateto .+1, :idoutcome = :idoutcome .+ 1);

        # Get a list of states, a list of actions, a list of models from test data 
        state_space_test, action_space_test,model_space_test = 
                                     get_state_action_model_space(test)
             # calculate rewards r_Test and transition probability trans_p_test on data set                       
        r_test,trans_p_test = calculate_reward_probability(test,
                             state_space_test, action_space_test,model_space_test)

        #Compute the policy and calculate the return                     
        returns = [] 
        time_record = []
        runtime_record = []
        
        
            
        #Calculate initial policy, and save q_{t,m}^{\pi}(s,a)
        #p_1 is the policy without model parameters; initial_policy is the policy with model parameters
        initial_policy,q_value = ini_extract_policy(T, state_space,action_space, 
                     model_space,r, trans_p, discount)
         #updated_policy = initial_policy
         #ml(model_lamda) :[T,model_space,state_space]
         ml= calculate_model_lamda(T,trans_p,model_space,action_space,
                                      state_space,initial_policy,ini_states)
            
            i = 1
            j =1
            done = false
            while !done
                alpha = 1/(i *j)
                updated_policy, new_q_value =  policy_next_iteration(initial_policy,
                                        q_value,ml,model_space,
                                      action_space,state_space,T,alpha,
                                             r, trans_p, discount)
                    
                value = kl_value(initial_policy,updated_policy,model_space,
                                      action_space,state_space,T)  
                
                
                if value < 1e-2
                    done = true
                    initial_policy = updated_policy
                else
                    initial_policy = updated_policy
                    q_value = new_q_value
                    ml= calculate_model_lamda(T,trans_p,model_space,action_space,
                                      state_space,initial_policy,ini_states)
                    
                end   
                j+= 1
            end
         end # end of the runtime
            
                update_state_value_test = zeros(Float64,(length(model_space_test), 
                                                   length(state_space_test),T+2))
               # result is the state values of states of all models at step 1
                result = evaluate_policy(T,initial_policy,r_test, trans_p_test, 
                         state_space_test,action_space_test, model_space_test,
                               update_state_value_test, discount )
             
                total_reward = calculate_total_reward(result,ini_states,model_space_test,state_space_test)
                
                
          
                push!(returns, total_reward)
                push!(time_record, T)
                push!(runtime_record,runtime/60)
       
        
         # write returns to file
        return_path = joinpath(@__DIR__,"resultfiles","return_Mirror_descent_$domain T_$initial_T.csv")
        return_value = DataFrame(Time=time_record,Return=returns)
        CSV.write(return_path, return_value) 
        
         #  write runtimes to file
        run_time_path = joinpath(@__DIR__,"resultfiles","time_Mirror_descent_$domain T_$initial_T.csv")
        run_time_value = DataFrame(Time=time_record,t_runtime=runtime_record)
        CSV.write(run_time_path, run_time_value) 
    end
       
    end 
end
    main()
