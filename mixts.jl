#UAI 2023
#Calculate the return of the MixTS algorithm

using Pkg
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("DataFramesMeta")
Pkg.add("StatsBase")
Pkg.add("Random")
Pkg.add("DataStructures")
Pkg.add("Distributions")

using Distributions
using DataStructures
using CSV
using DataFrames, DataFramesMeta
using CSV: File

using Random

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
                ini_states[j] = 0.0   
        end
    end
    return ini_states
end


# Caluate rewards and transition probabilities
function calculate_reward_probability(frame,state_space, action_space,model_space )
        
    state_size = length(state_space)
    action_size = length(action_space)
    model_size = length(model_space)
    
    # reward of going from state s to state s’ through action a.
    r = zeros((state_size, action_size,state_size,model_size))
    # transition probablity of (state, action, next state, model)
    p = zeros((state_size, action_size,state_size,model_size))
    # Given a model and a state, list of available actions an agent can take
    available_states = DefaultDict{Tuple{Int64,Int64,Int64}, Vector{Tuple{Int64,Float64}}}(Vector{Tuple{Int64,Float64}})
   
    for i in 1:size(frame,1)
        state_from = frame.idstatefrom[i]
        action = frame.idaction[i]
        state_to =  frame.idstateto[i]
        model =  frame.idoutcome[i]
        probability =  frame.probability[i]

        p[state_from, action, state_to, model] = frame.probability[i]
        r[state_from, action, state_to,model] = frame.reward[i]
        push!(available_states[(model,state_from,action)],(state_to,probability))
    end   
    
     # normalize transition probabilities
    p_n = zeros((state_size,action_size,state_size, model_size))
    for m in model_space
        for s_from in state_space
            for a in action_space
                sum_t = sum([p[s_from,a,s_next, m] for s_next in state_space])
                if sum_t > 1e-9
                    for s_next in state_space
                        p_n[s_from,a,s_next,m] = p[s_from,a,s_next,m]/sum_t  
                    end
                end
            end
        end
  end
              
    return  r,p_n,available_states
end


# get the best action for a state s in model m at time step t
#v_hat: value function
function get_max_action(m,s, r,p,t,state_space,action_space,
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
                d_hat[t,s,m] = get_max_action(m,s, r,p,t,state_space,action_space,
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

#Choose a next state based on the probability distribution of next states
#random_state:current state; random_action: the action taken at random_State
# t: current time step, available_states: its key is (model,state,action), and
# its key is a list of next available states a agent can reach
function choose_next_state(random_state,random_action,t,
                 available_states,model)
    
    #Get the list of tuples of (state,probability)
    list_of_tuples = available_states[(model,random_state,random_action)]
    len_list = length(list_of_tuples)
    
    if len_list == 0
        return "No available action!"
    end
    
    states=[]
    probabilities = Vector{Float64}()
   
    for i in  1:len_list 
        push!(states,list_of_tuples[i,1][1])
        push!(probabilities,list_of_tuples[i,1][2])
 
    end
    
    #Choose a next state based on probability distribution
    chosen_state =wsample(states,probabilities,1)[1]
    return chosen_state
end

#Generate a trajectory
#available_states: key is (model,state,action), value is the list of tuples(state,probability),
#policy_per_model: optimal deterministic policy for each model on training data
#policy_per_model[t,s,m]:the optimal action for state s of model m at step t
# chosen_model is the model selected at the begnning of a trajectory, and it is 
# the model of the environment, and it remains the same for a trajectory
function generate_trajectory( state_space,action_space,T,
                        available_states,policy_per_model, 
                            initial_state_distribution,chosen_model,
                      model_space_train, r_train,p_train,model_probability_train,
                          r_test,epsilon)
    # size of train models
    model_size = length(model_space_train)
    
    # store states of a trajectory at time steps 1..(T+1)
    trajectory_states = zeros(Int64,(T+2))   
    # store actions of a trajectory at time steps 1..T
    trajectory_actions = zeros(Int64,(T+2)) 
    #store model probabilities at time step 1,...,(T+1)
    model_probability = zeros(Float64,(model_size,T+2))
    probability = Vector{Float64}()
    probability_new =  Vector{Float64}()
    
    
    # save model probability, it is updated at each time step t
    for m in model_space_train
        push!(probability,model_probability_train[m])
        push!(probability_new,model_probability_train[m])
    end
    
    #Sample an initial state of the environment based on initial distribution of states
    ini = Vector{Float64}()
    for k in values(initial_state_distribution)
        push!(ini,k)
    end
    random_state = wsample(state_space,ini,1)[1]
    
    #Save the start state at time step 1
    trajectory_states[1] = random_state
    #Save model probabilities at T+1
    #weights = zeros((length(model_space_train)))
    
    #Sample a model of a policy, for the agent
    j = wsample(model_space_train,probability,1)[1]

    #Generate T+1 states and T actions, the first state is initialized above
    for t in 1:T
        #Select an action based on the optimal policy of model j
        random_action = policy_per_model[t,random_state,j]
        #Save the chosen action
        trajectory_actions[t] = random_action
       
        #Choose next state based on probability distribution of next states
        #chosen_model:the model selected by the environment
        random_state = choose_next_state(random_state,random_action,t,
                     available_states,chosen_model)
        #Save the chosen next state of the trajectory
        trajectory_states[t+1] = random_state
        
        #re is the reward Y_{t} in MixTs from the true environment
        re = r_test[trajectory_states[t],trajectory_actions[t],trajectory_states[t+1],
                    chosen_model]
        
        # Update model probability of an agent from the Bayes theorem in training data set
        for m in model_space_train
            state = trajectory_states[t]
            action = trajectory_actions[t]
            # P(Y_{t} | A_{t}; theta) = P(Y_{t},A_{t}) /P(A_{t})
            # calculate P(A_{t}
            P_A_t = sum([p_train[state,action,s,m ] for s in state_space])

            if P_A_t > 1e-9 
                p_y_a_t = 0.0
                for s in state_space
                    if abs(r_train[state,action,s,m] - re ) < 1e-9
                       #calculate P(Y_{t},A_{t}) 
                        p_y_a_t += p_train[state,action,s,m ]
                    end
                end
                        
                probability_new[m] = (p_y_a_t/P_A_t) * probability[m]
            else
                probability_new[m] = 0.0
            end  
        end 
        # Normalize model probabilities
        sum_weights = sum([probability_new[m] for m in model_space_train])
        #When the action is not taken in all training models, then model weights
        # for that state and action are undefined
        if abs(sum_weights) < 1e-9
            for m in model_space_train
                probability_new[m] =  1/ model_size
            end
        else
            for m in model_space_train
                probability_new[m] = probability_new[m] / sum_weights
            end
        end
        for m in model_space_train
            probability[m] = probability_new[m]
        end
    end              
    return trajectory_states, trajectory_actions,  probability
end

# Calculate expected reward for one trajectory
# r[state_from, action, state_to,model] , m is the chosen model
function calculate_reward_per_trajectory(trajectory_states, trajectory_actions,
           T,r,discount,m)
    
    # Save rewards of the trajectory of model m, m is the model selected by the environment
    reward = 0
    
    for t in 1:T
        s = trajectory_states[t]
        a = trajectory_actions[t]
        snext = trajectory_states[t+1]
        reward +=  (discount ^ (t-1)) * r[s,a,snext,m] 
    end  
    return reward
end


function main()
    domains = ['r']
    for domain in domains
        T = 50
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
       
        # Read a training file and offset relevant indices by one
        train_df = DataFrame(File(train_file)); #t_df: dataFrame of training.csv
        train = @transform(train_df, :idstatefrom = :idstatefrom .+1, :idaction = :idaction .+ 1,
                       :idstateto = :idstateto .+1, :idoutcome = :idoutcome .+ 1);
        state_space_train, action_space_train,model_space_train  = get_state_action_model_space(train)

        # calculate rewards r, and transition probability trans_p
        r_train,p_train,_= calculate_reward_probability(train,
                       state_space_train, action_space_train,model_space_train)
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
        # Get the initial distribution of states
        ini_df = DataFrame(File(ini_file)); #t_df: dataFrame of training.csv
        initial = @transform(ini_df, :idstate = :idstate .+1);
        initial_state_distribution = get_inital_state_distribution(initial, state_space_train)  
        
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
        test_df = DataFrame(File(test_file)); #t_df: dataFrame of training.csv
        test = @transform(test_df, :idstatefrom = :idstatefrom .+1, :idaction = :idaction .+ 1,
                       :idstateto = :idstateto .+1, :idoutcome = :idoutcome .+ 1);

        # Get a list of states, a list of actions, a list of models from test data 
        state_space_test, action_space_test,model_space_test = 
                                     get_state_action_model_space(test)
        # calculate rewards r_Test and transition probability trans_p_test on data set                       
        r_test,p_test,available_states_test =  calculate_reward_probability(test,
        state_space_test, action_space_test,model_space_test)
        
        # Size of state space and model space
        state_len_test = length(state_space_test)
        model_len_test = length(model_space_test)
        # Store the initial model probability     
        probability_test =  Vector{Float64}()
        model_probability_test = 1/ model_len_test 
                    
        for m in model_space_test
            push!(probability_test,model_probability_test)
        end
                 
        # Sample 1000 trajectories
        N = 1000                    

        #Compute the policy and calculate the return                     
        returns = [] 
        time_record = []
        all_returs= []
        tras = []
        while T> 1
    
            # state values at time step T+1 are set to 0                        
            update_state_value = zeros((length(model_space_test), length(state_space_test),T+2))
            # Extract the optimal policy for each model
            policy_per_model,_ = extract_policy_per_model(T, state_space_train,
                                action_space_train,model_space_train,update_state_value,
                                r_train, p_train, discount)

            # sum rewards of all trajectories
            sum_rewards_trajectories = 0.0      

            # Creat model weights over episodes
            model_probability = Vector{Float64}()
            model_size_train = length(model_space_train)
            model_probability_train = 1/ model_size_train
            for m in model_space_train
               push!(model_probability,model_probability_train)
            end
                
            Random.seed!(1000)
            for n in 1:N
                push!(tras,n) 
                 #Sample a model of the environment, hidden from the policy
                 # Randomly select a model and keep the model fixed for the trajectory
                model_selected_env = wsample(model_space_test,probability_test,1)[1]
               

                #Generate a trajectory, returns T+1 states and T actions
                 # w : save model probabilities at time step T+1
                trajectory_states, trajectory_actions,model_probability = generate_trajectory(state_space_test, action_space_test,
                    T,available_states_test,policy_per_model,
                        initial_state_distribution,model_selected_env,
                          model_space_train, 
                            r_train,p_train,model_probability,r_test,1e-9)
                 
                #calculate reward of the trajectory    
                reward = calculate_reward_per_trajectory(trajectory_states, 
                      trajectory_actions, T,r_test,discount,model_selected_env)
                # Save the return of each trajectory
                push!(all_returs, reward)
    
                #Sum rewards for N trajectories
                sum_rewards_trajectories += reward 
            end        
            temp_expected_reward = sum_rewards_trajectories / N
            push!(returns, temp_expected_reward)
            push!(time_record, T)
            
            T =  T-50
             #write all returns
        all_mp = joinpath(@__DIR__,"resultfiles","all_returns_mixts_$domain T_$T.csv")
        all_ab = DataFrame(Traj=tras,Return=all_returs)
        CSV.write(all_mp, all_ab)
        end
        #  write results to file
        mp = joinpath(@__DIR__,"resultfiles","mixts_$domain T_$initial_T.csv")
        ab = DataFrame(Time=time_record,Return=returns)
        CSV.write(mp, ab)
    end
end
    
main()
