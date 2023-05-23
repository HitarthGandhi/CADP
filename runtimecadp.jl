#UAI2023
#Calculate the runtime of the CADP algorithm

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


# get the best action for a state s in model m at time step t
#v_hat: value function
function get_init_max_action(m,s, r,p,t,state_space,action_space,
                   model_space, v_hat, discount)
    
    max_value =-Inf
    q_t_m = 0.0 # q_{t}^{m} (s_{t},a)
    max_action = 1
       
    for a in action_space
        # Exclude actions with probability 0.0
        pro = sum([p[s,a,next_s,m] for next_s in state_space])
        if pro > 1e-9
            # Calculate the further rewards
            further_reward = sum([p[s,a,next_s,m] * v_hat[m,next_s,t+1]
                              for next_s in state_space] )
            #Calculate the immediate rewards. This handles stochastic environment 
            immediate_reward = sum([r[s,a,s_next,m] * p[s,a,s_next,m]
                            for s_next in state_space])
            # reward + discount factor * sum of probablity of next state * state value
            q_t_m = immediate_reward +  discount  * further_reward 
     
            if q_t_m > max_value
                max_value = q_t_m
                max_action = a 
            end
        end
    end
    return max_action
 end

 # Extract a best policy for each model.
# p: transition probability, r:reward. update_state_value: state values of 
# all models at step t. update_state_value has the same values as v_hat
# This algorithm starts at step 1, instead of 0
function extract_policy_per_model(T, state_space,action_space,model_space,
                             update_state_value, r, p, discount)
    # d_hat save optimal actions taken at all states of different models for T steps
    d_hat = zeros(Int64,(T+1, length(state_space),length(model_space)))
    v_hat = update_state_value
    
    for m in model_space
        t = T # time step
        while t >= 1
            for s in state_space
                # get the optimal action for state s of model m at time step t
                d_hat[t,s,m] = get_init_max_action(m,s, r,p,t,state_space,action_space,
                           model_space, v_hat , discount)
            end
            for s in state_space
                #Calculate further rewards
                future_reward = sum([p[s,d_hat[t,s,m],s_next,m] * 
                                  v_hat[m,s_next,t+1]
                                  for s_next in state_space])
                # Calculate immediate rewards   
                immediate_reward = sum([r[s,d_hat[t,s,m],s_next,m] * p[s,d_hat[t,s,m],s_next,m]
                                   for s_next in state_space])  
                # update the state value of s in model m at time step t
                v_hat[m,s,t] = immediate_reward + discount  * future_reward   
            end                                 
            t=t-1 
        end
    end
    v = zeros((length(model_space), length(state_space)))
    for m in model_space
        for s in state_space
            # Return state values at time step 1
            v[m,s] = v_hat[m,s,1]
        end
    end
    # optimal value function for each state
    optimal_value = zeros((length(state_space)))
    for s in state_space
        for m in model_space
            optimal_value[s] += v[m,s]
        end
    end
    for s in state_space
        optimal_value[s] = optimal_value[s] / length(model_space)
    end
            
    return d_hat, optimal_value 
end




function calculate_model_lamda(T,p,model_space,action_space,state_space,initial_policy,ini_states)
    
    model_size = length(model_space)
    state_size = length(state_space)
      
    model_lamda = zeros((T+1,model_size,state_size))
    for m in model_space
        for s in state_space
                    model_lamda[1,m,s] = 1 / model_size * ini_states[s]
        end
    end
                
    for t in 2:T     
        for snext in state_space
            for m in model_space
                    for s in state_space
                        a = initial_policy[t-1,s,m]
                        model_lamda[t,m,snext] += p[s,a,snext,m] *
                               model_lamda[t-1,m,s]
                    end
            end
        end
    end

    return model_lamda  
end
 
 # get the optimal action for SWSU
function get_max_action(s,state_space,action_space,  model_space, update_state_value, 
                                     r,p,t,discount,ml)
                                    
    max_value = -Inf
    weighted_models_rewards = 0.0
    max_action = 1
    
    # w:weights of models
   w = zeros((length(model_space)))
    for m in  model_space
       w[m] = ml[t,m,s]  
    end
    
       
    for a in action_space
        # Make sure at least one model can take action a with non-zero probability
        pro = sum([p[s,a,next_s,m] for next_s in state_space
                   for m in model_space])
        if pro > 1e-15
            for m in model_space
                # Calculate the further rewards
                further_reward = sum([p[s,a,next_s,m] * update_state_value[m,next_s,t+1]
                                  for next_s in state_space] )
                #Calculate the immediate rewards. This handles stochastic environment 
                immediate_reward = sum([r[s,a,s_next,m] * p[s,a,s_next,m]
                                for s_next in state_space])
                # reward + discount factor * sum of probablity of next state * state value
                one_model_reward = immediate_reward +  discount  * further_reward 
                # sum values of different models
                weighted_models_rewards +=   one_model_reward * w[m]
            end
            if weighted_models_rewards > max_value
                max_value = weighted_models_rewards
                max_action = a 
            end   
            weighted_models_rewards = 0.0
        end
    end
    return max_action
end
   
# Extract a policy from training data.
# p is transition probability, r is reward. update_state_value: state values of 
# all models. This table is overwritten at every step. The final result of 
# this table is the state values of all models at step 1. 
# This algorithm starts at step 1, instead of 0
function extract_policy(T, state_space,action_space,model_space,update_state_value,
                   r, p, discount,ml)
    # pi of s at t, State -> action based on (8) in paper
    # store optimal actions taken for all states at time step 1..T
    pi_state = ones(Int64,(T+1, length(state_space)))
    policy = ones(Int64,(T+1, length(state_space),length(model_space)))
    
    t = T # time step
    while t>= 1
        for s in state_space
            # get the optimal action for state s at time step t
            pi_state[t,s] = get_max_action(s,state_space,action_space, model_space,
                                    update_state_value, r,p,t, discount,ml)
        end
        for m in model_space
            for s in state_space
                #Calculate further rewards
                future_reward = sum([p[s,pi_state[t,s],s_next,m] * 
                                  update_state_value[m,s_next, t+1]
                                  for s_next in state_space])
                # Calculate immediate rewards   
                immediate_reward = sum([r[s,pi_state[t,s],s_next,m] * p[s,pi_state[t,s],s_next,m]
                                   for s_next in state_space])  
                # update the state value of s in model m at step t
                update_state_value[m,s,t] = immediate_reward + discount  * future_reward   
            end
        end
                                          
        t=t-1 
    end   
    for s in state_space
        for t in 1:T
            for m in model_space
                policy[t,s,m]= pi_state[t,s]
            end
        end
    end
                
    return policy,pi_state
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
                #Calculate further reward
                future_reward = sum([p[s,policy[t,s],s_next,m]  * update_state_value[m,s_next,t+1]
                                 for s_next in state_space ])
                #Calculate the immediate reward   
                immediate_reward = sum([r[s,policy[t,s],s_next,m] * p[s,policy[t,s],s_next,m]
                                 for s_next in state_space])
                # update the state value of s in model m at step t                
                update_state_value[m,s,t] = immediate_reward + discount * future_reward
            end  
        end          
        t=t-1
    end
            
    v = zeros((length(model_space), length(state_space)))
    for m in model_space
        for s in state_space
            v[m,s] = update_state_value[m,s,1] 
        end
    end
        
    return v
end
# Total rewards of a policy. ini_states: initial distribution of states
# values_of_states: the state values of states of all models at step 1
function calculate_total_reward(values_of_states,ini_states,model_space,state_space)
    
    len_state = length(state_space)
    len_model = length(model_space)
    state_value_mean = zeros(length(state_space))
    total_reward = 0
    
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
    
    #domains = ['r','s','h','p']
    domains = ['r']
    for domain in domains
        #TS =[5,50,75,100,150]
        TS =[50]
        for T in TS
            timer= @elapsed begin
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
      
            update_value = zeros((length(model_space), length(state_space),T+2))

            # initial_policy,p_1 = ini_extract_policy(T, state_space,action_space, 
            #          model_space,update_value, r, trans_p, discount)
            initial_policy,p_1 = extract_policy_per_model(T, state_space,action_space, 
                     model_space,update_value, r, trans_p, discount)
                     
            
            #ml(model_lamda) :[T,model_space,state_space]
            ml= calculate_model_lamda(T,trans_p,model_space,action_space,
                                      state_space,initial_policy,ini_states)
           #-----------------------------------------------------
            done = false
            while !done
                update_1 = zeros((length(model_space), length(state_space),T+2))
                
                updated_policy,policy = extract_policy(T, state_space,action_space, 
                            model_space,update_1, r, trans_p, discount,ml)
                done =  isequal(initial_policy, updated_policy)
   
                if !done
                    initial_policy = updated_policy
                    ml= calculate_model_lamda(T,trans_p,model_space,action_space,
                                          state_space,updated_policy,ini_states)
                end

                if done
                    p_1 = policy
                end
            end  

        end  # The end of runtime  
             update_state_value_test = zeros((length(model_space_test), 
                                            length(state_space_test),T+2))
         
                # result is the state values of states of all models at step 1
             result = evaluate_policy(T,p_1,r_test, trans_p_test, 
                           state_space_test,action_space_test, model_space_test,
                                 update_state_value_test, discount )
                
            total_reward = calculate_total_reward(result,ini_states,model_space_test,state_space_test)
            
        
                # # write results to file
                mp = joinpath(@__DIR__,"resultfiles","cadp_time_$domain T_$T.csv")
                ab = DataFrame(T="$T",Time=timer/60)
                CSV.write(mp, ab)
        end
    end
        
 end

 main()