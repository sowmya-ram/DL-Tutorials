
using Flux
using MLUtils
using DataFrames
using CSV
using Dates
using Statistics
using JLD2
using Random
using StatsBase

Random.seed!(42)

anomaly_free = CSV.read("./data/anomaly-free/anomaly-free.csv", DataFrame, delim=';')
anomoly_1 = CSV.read("./data/other/1.csv", DataFrame, delim=';')
anomoly_2 = CSV.read("./data/other/2.csv", DataFrame, delim=';')
anomoly_3 = CSV.read("./data/other/3.csv", DataFrame, delim=';')
anomoly_4 = CSV.read("./data/valve1/0.csv", DataFrame, delim=';')
anomoly_5 = CSV.read("./data/valve1/1.csv", DataFrame, delim=';')
anomoly_6 = CSV.read("./data/valve1/2.csv", DataFrame   , delim=';')
anomoly_7 = CSV.read("./data/valve2/0.csv", DataFrame, delim=';')
anomoly_8 = CSV.read("./data/8.csv", DataFrame, delim=';')

anomoly_free_file = "./data/anomaly-free/anomaly-free.csv"
anomaly_all_files = [
    "./data/other/1.csv",
    "./data/other/2.csv",
    "./data/other/3.csv",
    "./data/valve1/0.csv",
    "./data/valve1/1.csv",
    "./data/valve1/2.csv",
    "./data/valve2/0.csv",
    "./data/8.csv"
]

function load_and_preprocess_data(normal_file, anomaly_files)
    println("Loading data...")
    
    normal_df = CSV.read(normal_file, DataFrame, delim=';')
    normal_df[!, :label] = zeros(Int, nrow(normal_df)) 
    
    # Load and combine anomaly data
    anomaly_dfs = DataFrame[]
    for file in anomaly_files
        df = CSV.read(file, DataFrame, delim=';')
        df[!, :label] = ones(Int, nrow(df)) 
       
        if "changepoint" in names(df)
            select!(df, Not(:changepoint))
        end
        push!(anomaly_dfs, df)
    end
    
    all_data = vcat(normal_df, anomaly_dfs...)
    
    feature_cols = [:Accelerometer1RMS, :Accelerometer2RMS, :Current, :Pressure, 
                   :Temperature, :Thermocouple, :Voltage, Symbol("Volume Flow RateRMS")]
    
    X = Matrix(all_data[:, feature_cols])' 
    y = all_data.label
    
    return X, y, feature_cols
end

# Normalization functions
function compute_normalization_stats(X_train)
    μ = mean(X_train, dims=2)
    σ = std(X_train, dims=2) .+ 1e-8  # Add small epsilon to avoid division by zero
    return μ, σ
end

function normalize_data(X, μ, σ)
    return (X .- μ) ./ σ
end

# Windowing function for sequence models
function create_windows(X, y, window_size)
    n_features, n_samples = size(X)
    if n_samples < window_size
        return nothing, nothing
    end
    
    n_windows = n_samples - window_size + 1
    X_windowed = zeros(Float32, n_features, window_size, n_windows)
    y_windowed = zeros(Int, n_windows)
    
    for i in 1:n_windows
        X_windowed[:, :, i] = X[:, i:i+window_size-1]
        # Use the label of the last time step in the window
        y_windowed[i] = y[i+window_size-1]
    end
    
    return X_windowed, y_windowed
end

# Model architectures
function create_feedforward_model(n_features)
    # Simple feedforward network with <1000 parameters
    # 8 features -> 64 -> 32 -> 1
    # Parameters: 8*64 + 64 + 64*32 + 32 + 32*1 + 1 = 512 + 64 + 2048 + 32 + 32 + 1 = 2689
    # Let's use smaller layers: 8 -> 32 -> 16 -> 1
    # Parameters: 8*32 + 32 + 32*16 + 16 + 16*1 + 1 = 256 + 32 + 512 + 16 + 16 + 1 = 833
    
    return Chain(
        Dense(n_features, 32, relu),
        Dense(32, 16, relu),
        Dense(16, 1, sigmoid)
    )
end

function create_lstm_model(n_features, window_size)
    # LSTM model with <1000 parameters
    # LSTM(8, 16) + Dense(16, 1)
    # LSTM parameters: 4 * (8*16 + 16*16 + 16) = 4 * (128 + 256 + 16) = 1600 (too many)
    # Let's use LSTM(8, 8) + Dense(8, 1)
    # Parameters: 4 * (8*8 + 8*8 + 8) + 8*1 + 1 = 4 * 136 + 9 = 544 + 9 = 553
    
    return Chain(
        LSTM(n_features, 8),
        Dense(8, 1, sigmoid)
    )
end

# Training function
function train_model(model, X_train, y_train, X_val, y_val; epochs=100, batch_size=128)
    # Convert labels to proper format
    y_train_flux = reshape(Float32.(y_train), 1, :)
    y_val_flux = reshape(Float32.(y_val), 1, :)
    
    # Create data loaders
    train_loader = DataLoader((X_train, y_train_flux), batchsize=batch_size, shuffle=true)
    
    # Optimizer with specified parameters
    opt = Flux.setup(AdamW(0.001, (0.9, 0.999), 0.001), model)  # lr=0.001, weight_decay=0.001
    
    # Training loop
    best_val_acc = 0.0
    best_model_state = nothing
    
    for epoch in 1:epochs
        # Training
        total_loss = 0.0
        num_batches = 0
        
        for (x_batch, y_batch) in train_loader
            # Ensure proper dimensions for LSTM if needed
            if ndims(x_batch) == 3
                # For LSTM: reset hidden state at the beginning of each batch
                Flux.reset!(model)
            end
            
            loss, grads = Flux.withgradient(model) do m
                ŷ = m(x_batch)
                Flux.logitbinarycrossentropy(ŷ, y_batch)
            end
            
            Flux.update!(opt, model, grads[1])
            
            total_loss += loss
            num_batches += 1
        end
        
        avg_loss = total_loss / num_batches
        
        # Validation
        if ndims(X_val) == 3
            Flux.reset!(model)
        end
        val_pred = model(X_val)
        val_pred_binary = val_pred .> 0.5
        val_acc = mean(val_pred_binary .== y_val_flux)
        
        # Calculate balanced accuracy
        y_val_bool = Bool.(y_val_flux[1, :])
        y_pred_bool = Bool.(val_pred_binary[1, :])
        
        tp = sum(y_val_bool .& y_pred_bool)
        tn = sum(.!y_val_bool .& .!y_pred_bool)
        fp = sum(.!y_val_bool .& y_pred_bool)
        fn = sum(y_val_bool .& .!y_pred_bool)
        
        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        balanced_acc = (sensitivity + specificity) / 2
        
        if balanced_acc > best_val_acc
            best_val_acc = balanced_acc
            best_model_state = deepcopy(Flux.state(model))
        end
        
        if epoch % 10 == 0 || epoch == 1
            println("Epoch $epoch: Loss = $(round(avg_loss, digits=4)), Val Acc = $(round(val_acc, digits=4)), Bal Acc = $(round(balanced_acc, digits=4))")
        end
    end
    
    # Restore best model
    Flux.loadmodel!(model, best_model_state)
    
    println("Best validation balanced accuracy: $(round(best_val_acc, digits=4))")
    return model, best_val_acc
end

# Balanced accuracy calculation function
function calculate_balanced_accuracy(y_true, y_pred)
    tp = sum(y_true .& y_pred)
    tn = sum(.!y_true .& .!y_pred)
    fp = sum(.!y_true .& y_pred)
    fn = sum(y_true .& .!y_pred)
    
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    
    return (sensitivity + specificity) / 2
end



X, y, feature_cols = load_and_preprocess_data(anomoly_free_file, anomaly_all_files)
n_features = size(X, 1)

println("Data shape: $(size(X))")
println("Labels distribution: Normal=$(sum(y.==0)), Anomaly=$(sum(y.==1))")
    


# Main training function
function main_training()
    
    # Load and preprocess data
    
    # Split data (80% train, 20% validation)
    n_samples = length(y)
    train_idx = 1:Int(floor(0.8 * n_samples))
    val_idx = (Int(floor(0.8 * n_samples)) + 1):n_samples
    
    X_train, X_val = X[:, train_idx], X[:, val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Compute normalization statistics
    μ, σ = compute_normalization_stats(X_train)
    
    # Normalize data
    X_train_norm = normalize_data(X_train, μ, σ)
    X_val_norm = normalize_data(X_val, μ, σ)
    
    results = Dict()
    
    # Train feedforward model
    println("\n=== Training Feedforward Model ===")
    ff_model = create_feedforward_model(n_features)
    println("Feedforward model parameters: $(sum(length, Flux.params(ff_model)))")
    
    trained_ff_model, ff_bal_acc = train_model(ff_model, X_train_norm, y_train, X_val_norm, y_val)
    results["feedforward"] = (ff_bal_acc, trained_ff_model, μ, σ, nothing)
    
    # Train LSTM models with different window sizes
    window_sizes = [30, 90, 270]
    
    for window_size in window_sizes
        println("\n=== Training LSTM Model (window_size=$window_size) ===")
        
        # Create windowed data
        X_train_windowed, y_train_windowed = create_windows(X_train_norm, y_train, window_size)
        X_val_windowed, y_val_windowed = create_windows(X_val_norm, y_val, window_size)
        
        if X_train_windowed === nothing
            println("Not enough data for window size $window_size")
            continue
        end
        
        println("Windowed training data shape: $(size(X_train_windowed))")
        
        lstm_model = create_lstm_model(n_features, window_size)
        println("LSTM model parameters: $(sum(length, Flux.params(lstm_model)))")
        
        trained_lstm_model, lstm_bal_acc = train_model(lstm_model, X_train_windowed, y_train_windowed, 
                                                      X_val_windowed, y_val_windowed)
        
        results["lstm_$window_size"] = (lstm_bal_acc, trained_lstm_model, μ, σ, window_size)
    end
    
    # Find best model
    best_model_name = ""
    best_bal_acc = 0.0
    
    for (name, (bal_acc, model, μ_val, σ_val, window_size)) in results
        println("$name: Balanced Accuracy = $(round(bal_acc, digits=4))")
        if bal_acc > best_bal_acc
            best_bal_acc = bal_acc
            best_model_name = name
        end
    end
    
    println("\nBest model: $best_model_name with balanced accuracy: $(round(best_bal_acc, digits=4))")
    
    # Save all models
    for (name, (bal_acc, model, μ_val, σ_val, window_size)) in results
        model_state = Flux.state(model)
        
        save_data = Dict(
            "trained_params" => model_state,
            "trained_st" => Dict(
                "μ" => μ_val,
                "σ" => σ_val,
                "window_size" => window_size,
                "model_type" => split(name, "_")[1],
                "n_features" => n_features,
                "balanced_accuracy" => bal_acc
            )
        )
        
        filename = "$(name)_model.jld2"
        JLD2.@save filename trained_params=save_data["trained_params"] trained_st=save_data["trained_st"]
        println("Saved model: $filename")
    end
    
    return results, best_model_name, best_bal_acc
end

# Run training
results, best_model_name, best_bal_acc = main_training()

println("\nTraining completed!")
println("Best model: $best_model_name")
println("Best balanced accuracy: $(round(best_bal_acc, digits=4))")