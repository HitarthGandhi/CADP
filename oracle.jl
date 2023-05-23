#UAI 2023
#Calculate the return of the Oracle algorithm

using Pkg
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("DataFramesMeta")

using CSV
using DataFrames, DataFramesMeta
using CSV: File


epsilon = 1e-9

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
# get the best action for given state s in model m at time step t
function get_max_action(s,m,state_space,action_space,  model_space, update_state_value, 
                                     r,p,t,discount)
    max_value = -Inf
    one_model_reward = 0.0
    max_action = 0
       
    for a in action_space
        pro = sum([p[s,a,next_s,m] for next_s in state_space])
        if (pro > epsilon)
                # Calculate the further rewards
            further_reward = sum([p[s,a,next_s,m] * update_state_value[m,next_s,t+1]
                              for next_s in state_space] )
            #Calculate the immediate rewards. This handles stochastic environment 
            immediate_reward = sum([r[s,a,s_next,m] * p[s,a,s_next,m]
                            for s_next in state_space])
            # reward + discount factor * sum of probablity of next state * state value
            one_model_reward = immediate_reward +  discount  * further_reward 
        end
 
        if one_model_reward > max_value
            max_value = one_model_reward
            max_action = a 
        end
    end

    return max_action
 end
# Extract a best policy from each model on test data.
# p is transition probability, r is reward. update_state_value: state values of 
# all models. This table is overwritten at every step. The final result of 
# this table is the state values of all models at step 1. 
# This algorithm starts at step 1, instead of 0
function extract_policy(T, state_space,action_space,model_space,update_state_value,
                   r, p, discount)
    # pi of s at t, State -> action based on (8) in paper
    # store optimal actions taken at all states of different models for T steps
    pi_state = zeros(Int64,(T+1, length(state_space),length(model_space)))
    
    for m in model_space
        t = T # time step
        while t>= 1
            for s in state_space
                # get the optimal action for state s of model m at time step t
                pi_state[t,s,m] = get_max_action(s,m,state_space,action_space, model_space,
                                        update_state_value, r,p,t, discount)
            end

            for s in state_space
                #Calculate further rewards
                future_reward = sum([p[s,pi_state[t,s,m],s_next,m] * update_state_value[m,s_next,t+1]
                                  for s_next in state_space])
                # Calculate immediate rewards   
                immediate_reward = sum([r[s,pi_state[t,s,m],s_next,m] * p[s,pi_state[t,s,m],s_next,m]
                                   for s_next in state_space])  
                # update the state value of s in model m at step t
                update_state_value[m,s,t] = immediate_reward + discount  * future_reward   
            end                             
            t=t-1
        end
    end
    v = zeros((length(model_space), length(state_space)))
    for m in model_space
        for s in state_space
            v[m,s] = update_state_value[m,s,1]  
        end
    end
            
    return pi_state, v
end

# Upper bound of a policy. ini_states: initial distribution of states
# values_of_states: the state values of states of all models at step 1,（m,s）
function calculate_upper_bound(values_of_states,ini_states,model_space,state_space)
    
    len_state = length(state_space)
    len_model = length(model_space)
    state_value_mean = zeros(length(state_space))
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
    all_mp = joinpath(@__DIR__,"resultfiles","all_returns_Oracle_.csv")
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
    
    #domains = ['p','s','r','h']  # Use riverswim data
    domains = ['r']
    for domain in domains
        T= 50
        initial_T = T
   
        action_space = []
        state_space = []
        model_space = []
         
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
        if (domain == 'i')
            discount_file = joinpath(@__DIR__,"domain","inventory","parameters.csv")
        end
        if (domain == 'h')
            discount_file = joinpath(@__DIR__,"domain","hiv","parameters.csv")
        end

         # Get the discount value
        discount = get_discount(discount_file)
       
         
         # Read test data
        if(domain == 'r')
                test_file = joinpath(@__DIR__,"domain","riverswim","test.csv")
        end
        if(domain == 's')
                test_file = joinpath(@__DIR__,"domain","population_small","test.csv")
        end
        if(domain == 'p')
                test_file = joinpath(@__DIR__,"domain","population","test.csv")  
        end
        if(domain == 'i')
                test_file = joinpath(@__DIR__,"domain","inventory","test.csv")
        end
        if(domain == 'h')
                test_file = joinpath(@__DIR__,"domain","hiv","test.csv")
        end

        # Read a training file and offset relevant indices by one
        test_df = DataFrame(File(test_file)); #t_df: dataFrame of training.csv
        test = @transform(test_df, :idstatefrom = :idstatefrom .+1, :idaction = :idaction .+ 1,
                               :idstateto = :idstateto .+1, :idoutcome = :idoutcome .+ 1);
        state_space, action_space,model_space = get_state_action_model_space(test)
        
        # calculate rewards r, and transition probability trans_p
        r,trans_p = calculate_reward_probability(test,state_space,
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
        if(domain == 'i')
           ini_file = joinpath(@__DIR__,"domain","inventory","initial.csv")
        end
        if(domain == 'h')
           ini_file = joinpath(@__DIR__,"domain","hiv","initial.csv")
        end
        
        # Get the initial distribution of states
        ini_df = DataFrame(File(ini_file)); #t_df: dataFrame of training.csv
        initial = @transform(ini_df, :idstate = :idstate .+1);
        ini_states = get_inital_state_distribution(initial, state_space)  

   
        returns = []
        time_record = []
        while T >=1
            
             # this table is used to update state values of all models
            update_state_value = zeros((length(model_space), length(state_space),T+2))
            
            policy, values_of_states = extract_policy(T, state_space,action_space, 
                         model_space,update_state_value, r, trans_p, discount)
            # calculate upper bound of discounted return
            upper_bound = calculate_upper_bound(values_of_states,ini_states,model_space,state_space)
            
          
            push!(returns, upper_bound)
            push!(time_record, T)
            T = T - 50
        end
        #  write results to file   
        mp = joinpath(@__DIR__,"resultfiles","Oracle_$domain T_$initial_T.csv" )  
        ab = DataFrame(Time=time_record,Return=returns)
        CSV.write(mp, ab)
    end
end
main()