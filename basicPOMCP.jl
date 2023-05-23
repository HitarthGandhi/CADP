#UAI 2023
#Compute the return and the runtime of the BasicPOMCP solver for POMDP planning

using Pkg
	Pkg.add("POMDPs")
	using POMDPs
	POMDPs.add_registry()
	Pkg.add("BasicPOMCP")
    Pkg.add("CSV")
    Pkg.add("DataFrames")
    Pkg.add("DataFramesMeta")
    Pkg.add("QuickPOMDPs")
    Pkg.add("POMDPModelTools")
    Pkg.add("POMDPModels")
    Pkg.add("POMDPSimulators")
    Pkg.add("Random")
    
    
    
    using Random
    using  POMDPModels, POMDPSimulators, BasicPOMCP
    using QuickPOMDPs
    using CSV
    using DataFrames, DataFramesMeta
    using CSV: File
    using POMDPModelTools: SparseCat
    using POMDPModelTools
    using QuickPOMDPs: DiscreteExplicitPOMDP
    

    # QuickPOMDPs --> Discrete Explicit Interface
    #https://juliapomdp.github.io/QuickPOMDPs.jl/dev/discrete_explicit/#Exampl

    runtime= @elapsed begin
    
          
        # Domain riverswim
        discountpath= joinpath(@__DIR__,"domain","riverswim",
        "parameters.csv");
        testpath = joinpath(@__DIR__,"domain","riverswim",
        "test.csv");
        initialpath = joinpath(@__DIR__,"domain","riverswim",
        "initial.csv");
        
    
        #=
        # Domain inventory
        discountpath= joinpath(@__DIR__,"domain","inventory",
        "parameters.csv");
        testgpath = joinpath(@__DIR__,"domain","inventory",
        "test.csv");
        initialpath = joinpath(@__DIR__,"domain","inventory",
        "initial.csv");
        =# 

        #=
        # Domain hiv
        discountpath= joinpath(@__DIR__,"domain","hiv",
        "parameters.csv");
        testpath = joinpath(@__DIR__,"domain","hiv",
        "test.csv");
        initialpath = joinpath(@__DIR__,"domain","hiv",
        "initial.csv");
        =#
    
         #= 
        # Domain population
        discountpath= joinpath(@__DIR__,"domain","population",
        "parameters.csv");
        testpath = joinpath(@__DIR__,"domain","population",
        "test.csv");
        initialpath = joinpath(@__DIR__,"domain","population",
        "initial.csv");
        =#
    
        #=   
        # Domain population_small
        discountpath= joinpath(@__DIR__,"domain","population_small",
        "parameters.csv");
        testpath = joinpath(@__DIR__,"domain","population_small",
        "test.csv");
        initialpath = joinpath(@__DIR__,"domain","population_small",
        "initial.csv");
        =#

    # Read discount factor
    discount = DataFrame(File(discountpath))[1,2]; #single value 0.9
    
    # Read a training file and offset relevant indices by one
    t_df = DataFrame(File(testpath)); #t_df: dataFrame of training.csv
    t_dfone = @transform(t_df, :idstatefrom = :idstatefrom .+1, :idaction = :idaction .+ 1,
               :idstateto = :idstateto .+1, :idoutcome = :idoutcome .+ 1);


    statecount = max(maximum(t_dfone.idstatefrom), maximum(t_dfone.idstateto))
    actioncount = maximum(t_dfone.idaction)
    modelcount = maximum(t_dfone.idoutcome)
    lambda = 1 / modelcount
    statemodelcount = statecount * modelcount

    # read initial distribution over states
    i_df = DataFrame(File(initialpath)); #i_df:  dataFrame of initial.csv
    i_dfOne=@transform(i_df, :idstate= :idstate .+1);
    

    # Use SparseCat(values, probabilities)
    #values is an iterable object containing the possible values (can be of any type) 
    #in the distribution that have nonzero probability. probabilities is an iterable object 
    #that contains the associated probabilities.

    len_initial_states = size(i_dfOne,1) 
    values = [n for n=1:statemodelcount]
    probabilities = [0.0*n for n=1:statemodelcount]
    for s in 1: len_initial_states
        for m in 1:modelcount
            probabilities[ (i_dfOne.idstate[s]-1)*modelcount + m] = i_dfOne.probability[s]*lambda
        end
    end

    #intial belief
    b0= SparseCat(values, probabilities)
    

    # state space S 1: (1,1); 2:(1,2);.... last:(|S|,|M|)
    # state number: sn= ceiling(n/modelcount)
    # model number: mn = n-(sn-1)*modelcount
    S = 1:statemodelcount

    #Action space A
    A = 1:actioncount
    
    # Observation space O = {1,2,...,|S|},S is the state space of MMDP
    O = 1:statecount
   

    #T(s,a,s ) is the probability of transitioning to state s'from state s after taking action aa.
    #Construct transition probaility T, |S|x|A|x|S|, T[s, a, s'] = p(s'|a,s) for POMDP
    #s1: next state, s: current state 
     function T(s1,a,s2) 
        s_1 = ceil(s1/modelcount );
        s_2 = ceil(s2/modelcount) ;
        m1 = s1 -(s_1-1)*modelcount;
        m2 = s2 -(s_2-1)*modelcount;

        if m1 != m2
            return 0.0
        else 
           for i in 1:size(t_dfone,1)
               if t_dfone.idstateto[i] == s_2 && t_dfone.idstatefrom[i] == s_1 && t_dfone.idoutcome[i] == m1 && t_dfone.idaction[i]== a
                 return t_dfone.probability[i]
               end
            end
            return 0.0  # No p(s1|s,a) in model m
        end
     end
    
     # Z::Function: Observation probability distribution function; O(a, s', o) is the 
     # probability of receiving observation oo when state s'is reached after action a.
     function Z(a, s, o)
        s_1 = ceil(s/modelcount );

        if s_1 == o
            return 1.0
        else 
            return 0.0
        end
    end

    # Calculate r^m(s,a)
    function r(s,a,m)
        sumr = 0.0
        for i in 1:size(t_dfone,1)
            if t_dfone.idstatefrom[i] == s && t_dfone.idaction[i]== a && t_dfone.idoutcome[i] == m 
              sumr = sumr + t_dfone.reward[i] *t_dfone.probability[i]
            end
         end 
        if abs(sumr) > 1e-15
            return sumr 
        else 
             return 0  
        end
    end
    # R::Function: Reward function; R(s,a)R(s,a) is the reward for taking action aa in state ss.
    function R(s, a)
        s1 = ceil(s/modelcount );
        m1 = s -(s1-1)*modelcount;
        return r(s1,a,m1)
    end

    m = DiscreteExplicitPOMDP(S,A,O,T,Z,R,discount,b0)
   rng = MersenneTwister(1);
   #rng = 5 

   # The Partially Observable Monte Carlo Planning (POMCP) online solver for POMDPs.jl.
   # Described inSilver, D., & Veness, J. (2010). 
   # Monte-Carlo Planning in Large POMDPs. In Advances in neural information processing systems (pp. 2164–2172).
   # POMCPSolver implementation is on https://github.com/JuliaPOMDP/BasicPOMCP.jl
   solver = POMCPSolver( tree_queries=50)
   policy = solve(solver, m)
end # the end of run time
   
   up = updater(policy)
   return_ave = 0.0 
   num_trajectory = 50
   timesteps = 10
   returns_trajectory = []
   trajectories = []
   Random.seed!(1000)
   for iteration in 1:50
       r_total = 0.0
       s = rand(initialstate(m))
       b = initialize_belief(up,b0)
       d = discount
       for i in 1:timesteps
            a = action(policy, b)
            sp = rand(transition(m,s,a))
       local o = rand(observation(m,s,a,sp))
       local r =reward(m,s,a,sp,o)
            s= sp
            r_total += d*r
            d = d *discount
            b= update(up,b,a,o)
       end

       push!(returns_trajectory, r_total)
       push!(trajectories, iteration)
       global return_ave += r_total
   
   end

   ave = return_ave /num_trajectory
     

    # Results for domain riverswim
    #  write returns to file
    return_path = joinpath(@__DIR__,"resultfiles","return_pomcp_50_riverswim.csv")
    return_sampled = DataFrame(Time=timesteps,Return=ave)
    CSV.write(return_path, return_sampled)
    # write runtimes to file
    runtime_path = joinpath(@__DIR__,"resultfiles","time_pomcp_50_riverswim.csv")
    runtime = DataFrame(Time=timesteps,Runtime=runtime/60)
    CSV.write(runtime_path, runtime)

    #  write returns of all trajectories to file
    returns_path = joinpath(@__DIR__,"resultfiles","all_returns_pomcp_50_riverswim.csv")
    returns_sampled = DataFrame(Time=trajectories,Return=returns_trajectory)
    CSV.write(returns_path, returns_sampled)
   
    #=
    # Results for domain inventory
    #  write returns to file
    return_path = joinpath(@__DIR__,"resultfiles","return_pomcp_50_inventory.csv")
    return_sampled = DataFrame(Time=timesteps,Return=ave)
    CSV.write(return_path, return_sampled)
    # write runtimes to file
    runtime_path = joinpath(@__DIR__,"resultfiles","time_pomcp_50_inventory.csv")
    runtime = DataFrame(Time=timesteps,Runtime=runtime/60)
    CSV.write(runtime_path, runtime)

    #  write returns of all trajectories to file
    returns_path = joinpath(@__DIR__,"resultfiles","all_returns_pomcp_50_inventory.csv")
    returns_sampled = DataFrame(Time=trajectories,Return=returns_trajectory)
    CSV.write(returns_path, returns_sampled)
    =#


    #=
    # Results for domain hiv
    #  write returns to file
    return_path = joinpath(@__DIR__,"resultfiles","return_pomcp_10_hiv.csv")
    return_sampled = DataFrame(Time=timesteps,Return=ave)
    CSV.write(return_path, return_sampled)
    # write runtimes to file
    runtime_path = joinpath(@__DIR__,"resultfiles","time_pomcp_10_hiv.csv")
    runtime = DataFrame(Time=timesteps,Runtime=runtime/60)
    CSV.write(runtime_path, runtime)
    #  write returns of all trajectories to file
    returns_path = joinpath(@__DIR__,"resultfiles","all_returns_pomcp_10_hiv.csv")
    returns_sampled = DataFrame(Time=trajectories,Return=returns_trajectory)
    CSV.write(returns_path, returns_sampled)
    =# 
    
    #=
    # Results for domain population
    #  write returns to file
    return_path = joinpath(@__DIR__,"resultfiles","return_pomcp_50_population.csv")
    return_sampled = DataFrame(Time=timesteps,Return=ave)
    CSV.write(return_path, return_sampled)
    # write runtimes to file
    runtime_path = joinpath(@__DIR__,"resultfiles","time_pomcp_50_population.csv")
    runtime = DataFrame(Time=timesteps,Runtime=runtime/60)
    CSV.write(runtime_path, runtime)
    =#

    #= 
    # Results for domain population_small
    #  write returns to file
    return_path = joinpath(@__DIR__,"resultfiles","return_pomcp_50_population_small.csv")
    return_sampled = DataFrame(Time=timesteps,Return=ave)
    CSV.write(return_path, return_sampled)
    # write runtimes to file
    runtime_path = joinpath(@__DIR__,"resultfiles","time_pomcp_50_population_small.csv")
    runtime = DataFrame(Time=timesteps,Runtime=runtime/60)
    CSV.write(runtime_path, runtime)
    =#