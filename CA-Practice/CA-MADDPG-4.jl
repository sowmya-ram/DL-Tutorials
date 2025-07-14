using Flux
using Flux: params
using Random
using Statistics
using Plots
using LinearAlgebra

# Define structures
mutable struct Agent
    actor::Chain
    critic::Chain
    target_actor::Chain
    target_critic::Chain
    actor_optimizer
    critic_optimizer
    γ::Float32
    τ::Float32
end

mutable struct ReplayBuffer
    max_size::Int
    states::Vector{Any}
    actions::Vector{Any}
    rewards::Vector{Any}
    next_states::Vector{Any}
    ptr::Int
    size::Int
end

mutable struct MADDPG
    agents::Vector{Agent}
    replay_buffer::ReplayBuffer
    batch_size::Int
    update_freq::Int
end

# Replay buffer functions
function ReplayBuffer(max_size::Int)
    ReplayBuffer(max_size, [], [], [], [], 1, 0)
end

function add_experience!(buffer::ReplayBuffer, state, action, reward, next_state)
    if buffer.size < buffer.max_size
        push!(buffer.states, state)
        push!(buffer.actions, action)
        push!(buffer.rewards, reward)
        push!(buffer.next_states, next_state)
        buffer.size += 1
    else
        buffer.states[buffer.ptr] = state
        buffer.actions[buffer.ptr] = action
        buffer.rewards[buffer.ptr] = reward
        buffer.next_states[buffer.ptr] = next_state
    end
    buffer.ptr = mod1(buffer.ptr + 1, buffer.max_size)
end

function sample_minibatch(buffer::ReplayBuffer, batch_size::Int)
    idxs = rand(1:buffer.size, min(batch_size, buffer.size))
    return (
        [buffer.states[i] for i in idxs],
        [buffer.actions[i] for i in idxs],
        [buffer.rewards[i] for i in idxs],
        [buffer.next_states[i] for i in idxs]
    )
end

# Neural network creation
function create_actor(state_dim::Int, action_dim::Int)
    Chain(
        Dense(state_dim, 16, relu),
        Dense(16, 16, relu),
        Dense(16, action_dim, tanh)
    )
end

function create_critic(state_dim::Int, action_dim::Int, n_agents::Int)
    total_action_dim = action_dim * n_agents
    Chain(
        Dense(state_dim + total_action_dim, 16, relu),
        Dense(16, 16, relu),
        Dense(16, 1)
    )
end

# Count parameters
function count_parameters(model::Chain)
    sum(length, params(model))
end

# Soft update for target networks
function soft_update!(target, source, τ)
    for (t, s) in zip(params(target), params(source))
        t .= τ .* s .+ (1 - τ) .* t
    end
end

# Initialize MADDPG
function MADDPG(state_dim::Int, action_dim::Int, n_agents::Int; 
                γ::Float32=0.95f0, τ::Float32=0.01f0, 
                lr_actor=0.001f0, lr_critic=0.001f0,
                buffer_size=1000000, batch_size=1024, update_freq=100)
    
    agents = Agent[]
    for i in 1:n_agents
        actor = create_actor(state_dim, action_dim)
        critic = create_critic(state_dim, action_dim, n_agents)
        target_actor = deepcopy(actor)
        target_critic = deepcopy(critic)
        
        agent = Agent(
            actor,
            critic,
            target_actor,
            target_critic,
            ADAM(lr_actor),
            ADAM(lr_critic),
            γ,
            τ
        )
        push!(agents, agent)
        
        println("Agent $i: Actor parameters = $(count_parameters(actor)), Critic parameters = $(count_parameters(critic))")
    end
    
    replay_buffer = ReplayBuffer(buffer_size)
    
    MADDPG(agents, replay_buffer, batch_size, update_freq)
end

# Get actions
function get_actions(maddpg::MADDPG, states, exploration_noise=true)
    actions = []
    for (i, agent) in enumerate(maddpg.agents)
        state = states[i]
        action = agent.actor(state)
        if exploration_noise
            action = action .+ randn(Float32, size(action)) * 0.1f0
        end
        push!(actions, clamp.(action, -1.0f0, 1.0f0))
    end
    return actions
end

# Update function
function update!(maddpg::MADDPG, states, actions, rewards, next_states)
    if maddpg.replay_buffer.size < maddpg.batch_size
        return
    end
    
    batch = sample_minibatch(maddpg.replay_buffer, maddpg.batch_size)
    batch_states, batch_actions, batch_rewards, batch_next_states = batch
    
    for (i, agent) in enumerate(maddpg.agents)
        # Update critic
        all_actions = hcat([batch_actions[j][i] for j in 1:length(batch_actions)]...)
        all_next_actions = [agent.target_actor(batch_next_states[j][i]) for j in 1:length(batch_next_states)]
        all_next_actions = hcat(all_next_actions...)
        
        next_q = agent.target_critic(vcat(hcat(batch_next_states...), all_next_actions))
        target_q = batch_rewards[i] .+ agent.γ * next_q
        
        critic_loss = mean((agent.critic(vcat(hcat(batch_states...), all_actions)) .- target_q).^2)
        
        Flux.back!(critic_loss)
        Flux.update!(agent.critic_optimizer, params(agent.critic))
        
        # Update actor
        actor_actions = [agent.actor(batch_states[j][i]) for j in 1:length(batch_states)]
        all_actions_with_actor = copy(all_actions)
        all_actions_with_actor[:,i] = hcat(actor_actions...)
        
        actor_loss = -mean(agent.critic(vcat(hcat(batch_states...), all_actions_with_actor)))
        
        Flux.back!(actor_loss)
        Flux.update!(agent.actor_optimizer, params(agent.actor))
        
        # Update target networks
        soft_update!(agent.target_actor, agent.actor, agent.τ)
        soft_update!(agent.target_critic, agent.critic, agent.τ)
    end
end

# Cooperative communication environment
mutable struct CoopCommEnvironment
    state_dim::Int
    action_dim::Int
    n_agents::Int
    landmarks::Vector{Vector{Float32}}
    target::Int
    agent_positions::Vector{Vector{Float32}}
end

function CoopCommEnvironment(n_agents::Int)
    state_dim = 6
    action_dim = 2
    landmarks = [rand(Float32, 2) * 10 for _ in 1:3]
    target = rand(1:3)
    agent_positions = [rand(Float32, 2) * 10 for _ in 1:n_agents]
    CoopCommEnvironment(state_dim, action_dim, n_agents, landmarks, target, agent_positions)
end

function reset!(env::CoopCommEnvironment)
    env.landmarks = [rand(Float32, 2) * 10 for _ in 1:3]
    env.target = rand(1:3)
    env.agent_positions = [rand(Float32, 2) * 10 for _ in 1:env.n_agents]
    states = [vcat(env.agent_positions[i], env.landmarks[env.target], env.agent_positions[mod1(i+1, env.n_agents)]) for i in 1:env.n_agents]
    return states
end

function step!(env::CoopCommEnvironment, actions)
    for i in 1:env.n_agents
        env.agent_positions[i] .+= actions[i] * 0.1
        env.agent_positions[i] = clamp.(env.agent_positions[i], 0.0f0, 10.0f0)
    end
    rewards = [-norm(env.agent_positions[i] - env.landmarks[env.target]) for i in 1:env.n_agents]
    next_states = [vcat(env.agent_positions[i], env.landmarks[env.target], env.agent_positions[mod1(i+1, env.n_agents)]) for i in 1:env.n_agents]
    done = any(norm(env.agent_positions[i] - env.landmarks[env.target]) < 0.5 for i in 1:env.n_agents)
    return next_states, rewards, done
end

# Training function with metrics
function train!(maddpg::MADDPG, env::CoopCommEnvironment, max_episodes::Int, max_steps::Int)
    episode_rewards = Float32[]
    target_reaches = Float32[]
    avg_distances = Float32[]
    
    for episode in 1:max_episodes
        state = reset!(env)
        episode_reward = zeros(Float32, env.n_agents)
        min_distance = Inf32
        
        for t in 1:max_steps
            actions = get_actions(maddpg, state, true)
            next_state, rewards, done = step!(env, actions)
            
            add_experience!(maddpg.replay_buffer, state, actions, rewards, next_state)
            
            if maddpg.replay_buffer.size >= maddpg.batch_size && t % maddpg.update_freq == 0
                update!(maddpg, state, actions, rewards, next_state)
            end
            
            episode_reward .+= rewards
            min_distance = min(min_distance, minimum(norm(env.agent_positions[i] - env.landmarks[env.target]) for i in 1:env.n_agents))
            state = next_state
            
            if done
                break
            end
        end
    
        push!(episode_rewards, mean(episode_reward))
        push!(target_reaches, min_distance < 0.5 ? 1.0f0 : 0.0f0)
        push!(avg_distances, min_distance)
        
        if episode % 100 == 0
            println("Episode $episode: Avg Reward = $(mean(episode_reward)), Target Reach = $(target_reaches[end]), Avg Distance = $min_distance")
        end
    end
    
    return episode_rewards, target_reaches, avg_distances
end

# Evaluation function (fixed n_agents -> env.n_agents)
function evaluate!(maddpg::MADDPG, env::CoopCommEnvironment, n_eval_episodes::Int, max_steps::Int)
    target_reach_count = 0
    distances = Float32[]
    trajectories = [Vector{Vector{Float32}}[] for _ in 1:env.n_agents]  # Fixed n_agents
    
    for episode in 1:n_eval_episodes
        state = reset!(env)
        min_distance = Inf32
        episode_trajectories = [Vector{Vector{Float32}}() for _ in 1:env.n_agents]  # Fixed n_agents
        
        for t in 1:max_steps
            for i in 1:env.n_agents  # Fixed n_agents
                push!(episode_trajectories[i], copy(env.agent_positions[i]))
            end
            actions = get_actions(maddpg, state, false)
            next_state, rewards, done = step!(env, actions)
            min_distance = min(min_distance, minimum(norm(env.agent_positions[i] - env.landmarks[env.target]) for i in 1:env.n_agents))
            state = next_state
            if done
                break
            end
        end
        
        if min_distance < 0.5
            target_reach_count += 1
        end
        push!(distances, min_distance)
        for i in 1:env.n_agents  # Fixed n_agents
            push!(trajectories[i], episode_trajectories[i])
        end
    end
    
    target_reach_percentage = target_reach_count / n_eval_episodes * 100
    avg_distance = mean(distances)
    
    println("Evaluation over $n_eval_episodes episodes:")
    println("Target Reach Percentage: $target_reach_percentage%")
    println("Average Distance to Target: $avg_distance")
    
    # Plot sample trajectories (first episode)
    p = plot(title="Agent Trajectories (Episode 1)")
    for i in 1:env.n_agents  # Fixed n_agents
        traj = trajectories[i][1]
        plot!([t[1] for t in traj], [t[2] for t in traj], label="Agent $i")
    end
    scatter!([env.landmarks[env.target][1]], [env.landmarks[env.target][2]], label="Target", markersize=10)
    savefig("trajectories5.png")
    
    return target_reach_percentage, avg_distance
end

# Visualization function
function plot_metrics(episode_rewards, target_reaches, avg_distances)
    p1 = plot(episode_rewards, label="Average Reward", title="Training Progress")
    p2 = plot(moving_average(target_reaches, 100) * 100, label="Target Reach %", title="Target Reach Percentage")
    p3 = plot(moving_average(avg_distances, 100), label="Avg Distance", title="Average Distance to Target")
    plot(p1, p2, p3, layout=(3,1), size=(800,600))
end

function moving_average(data, window)
    return [mean(data[max(1,i-window+1):i]) for i in 1:length(data)]
end

# Main function
function main()
    Random.seed!(123)
    
    n_agents = 4
    state_dim = 6
    action_dim = 2
    max_episodes = 5000
    max_steps = 25
    n_eval_episodes = 100
    
    env = CoopCommEnvironment(n_agents)
    maddpg = MADDPG(state_dim, action_dim, n_agents)
    
    # Train
    rewards, target_reaches, avg_distances = train!(maddpg, env, max_episodes, max_steps)
    
    # Evaluate
    target_reach_percentage, avg_distance = evaluate!(maddpg, env, n_eval_episodes, max_steps)
    
    # Plot results
    plot_metrics(rewards, target_reaches, avg_distances)
    savefig("training_metrics5.png")
    
    
    return rewards, target_reaches, avg_distances
end

main()