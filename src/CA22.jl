using CSV
using DataFrames
using Flux
using Flux: onehotbatch, onecold
using Statistics
using Dates
using Random
using StatsBase
using MLUtils
using JLD2

Random.seed!(42)

# File paths
anomaly_free_file = "./data/anomaly-free/anomaly-free.csv"
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

# Hyperparameters
window_sizes = [30, 90, 270]
max_params = 1000
epochs = 100
batch_size = 128

# Load and preprocess data
function load_and_preprocess_data(anomaly_free_file, anomaly_files)
    # Load datasets
    df_normal = CSV.read(anomaly_free_file, DataFrame, delim=';', dateformat="yyyy-mm-dd HH:MM:SS")
    df_normal.anomaly = zeros(Float32, nrow(df_normal))
    dfs_anomaly = [CSV.read(file, DataFrame, delim=';', dateformat="yyyy-mm-dd HH:MM:SS") for file in anomaly_files]
    
    # Combine datasets
    df = vcat(df_normal, dfs_anomaly..., cols=:union)
    
    # Sort by datetime
    sort!(df, :datetime)
    
    # Select features (exclude datetime, changepoint, dataset_id)
    feature_cols = [col for col in names(df) if !(col in ["datetime", "changepoint", "anomaly"])]
    X = Matrix{Float32}(df[:, feature_cols])
    y = Float32.(df.anomaly)
    
    # Normalize features
    X_mean = mean(X, dims=1)
    X_std = std(X, dims=1)
    X_std[X_std .== 0] .= 1f0  # Avoid division by zero
    X_normalized = (X .- X_mean) ./ X_std
    
    return X_normalized, y, X_mean, X_std, feature_cols
end

# Create sequences
function create_sequences(X, y, window_size::Int; stride=1)
    sequences_X = []
    sequences_y = []
    
    for i in 1:stride:(size(X, 1) - window_size + 1)
        seq_X = X[i:(i + window_size - 1), :]'  # Shape: (features, window_size)
        seq_y = y[i + window_size - 1]
        push!(sequences_X, seq_X)
        push!(sequences_y, seq_y)
    end
    
    return cat(sequences_X..., dims=3), vcat(sequences_y...)  # Shape: (features, window_size, samples)
end

# Build LSTM model
function create_lstm_model(input_size::Int, window_size::Int; max_params::Int=1000)
    hidden_size = 8  # Start with small size to stay under parameter limit
    
    # Calculate parameters
    lstm_params = 4 * hidden_size * (input_size + hidden_size + 1)
    dense_params = hidden_size + 1
    total_params = lstm_params + dense_params
    
    while total_params > max_params && hidden_size > 4
        hidden_size -= 1
        lstm_params = 4 * hidden_size * (input_size + hidden_size + 1)
        dense_params = hidden_size + 1
        total_params = lstm_params + dense_params
    end
    
    @assert total_params <= max_params "Cannot create model with <= $max_params parameters"
    println("Model architecture: LSTM($input_size, $hidden_size) + Dense($hidden_size, 1)")
    println("Total parameters: $total_params")
    
    model = Chain(
        LSTM(input_size => hidden_size),
        Dense(hidden_size => 1, sigmoid)
    )
    
    return model
end

# Balanced accuracy
function balanced_accuracy(y_true, y_pred)
    tp = sum((y_true .== 1) .& (y_pred .== 1))
    fp = sum((y_true .== 0) .& (y_pred .== 1))
    tn = sum((y_true .== 0) .& (y_pred .== 0))
    fn = sum((y_true .== 1) .& (y_pred .== 0))
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    return (sensitivity + specificity) / 2
end

# Training function
function train_model(model, train_data, val_data; epochs=100, batch_size=128)
    opt = AdamW(0.001f0, (0.9f0, 0.999f0), 0.001f0)
    loss(x, y) = Flux.binarycrossentropy(model(x), y)
    
    best_val_acc = 0.0
    best_model_state = nothing
    patience = 20
    patience_counter = 0
    
    for epoch in 1:epochs
        # Train
        train_loss = 0.0f0
        for (x, y) in train_data
            Flux.reset!(model)
            loss_val, grads = Flux.withgradient(model) do m
                mean(Flux.binarycrossentropy.(m(x), y))
            end
            Flux.update!(opt, model, grads[1])
            train_loss += loss_val
        end
        train_loss /= length(train_data)
        
        # Validation
        if epoch % 5 == 0 || epoch == epochs
            val_acc, val_bal_acc = evaluate_model(model, val_data)
            println("Epoch $epoch: Train Loss = $(round(train_loss, digits=4)), Val Acc = $(round(val_acc, digits=4)), Val Bal Acc = $(round(val_bal_acc, digits=4))")
            
            if val_bal_acc > best_val_acc
                best_val_acc = val_bal_acc
                best_model_state = deepcopy(Flux.state(model))
                patience_counter = 0
            else
                patience_counter += 1
                if patience_counter >= patience
                    println("Early stopping at epoch $epoch")
                    break
                end
            end
        end
    end
    
    return best_model_state, best_val_acc
end

# Evaluation function
function evaluate_model(model, data)
    predictions = Float32[]
    true_labels = Float32[]
    
    for (x, y) in data
        Flux.reset!(model)
        ŷ = model(x)[1]
        push!(predictions, ŷ >= 0.5f0 ? 1.0f0 : 0.0f0)
        push!(true_labels, y)
    end
    
    accuracy = mean(predictions .== true_labels)
    bal_acc = balanced_accuracy(true_labels, predictions)
    
    return accuracy, bal_acc
end

# Balanced accuracy function for testing
function bal_acc(model_path, window_size, X_mean, X_std, feature_cols, test_x, test_y)
    # Load model
    JLD2.@load model_path trained_params
    model = create_lstm_model(length(feature_cols), window_size)
    Flux.loadmodel!(model, trained_params)
    
    # Preprocess test data
    X_test = Matrix{Float32}(test_x[:, feature_cols])
    X_test_normalized = (X_test .- X_mean) ./ X_std
    X_test_win, y_test_win = create_sequences(X_test_normalized, Float32.(test_y), window_size)
    
    # Evaluate
    predictions = [model(X_test_win[:, :, i])[1] >= 0.5f0 ? 1.0f0 : 0.0f0 for i in 1:size(X_test_win, 3)]
    return balanced_accuracy(y_test_win, predictions)
end

# Main execution
# Load and preprocess data
X, y, X_mean, X_std, feature_cols = load_and_preprocess_data(anomaly_free_file, anomaly_all_files)
println("Total samples: $(size(X, 1)), Features: $(length(feature_cols))")
println("Class distribution: ", countmap(y))

best_models = Dict()

for window_size in window_sizes
    println("\nTraining model with window size: $window_size")
    
    # Create sequences
    X_seq, y_seq = create_sequences(X, y, window_size)
    data_pairs = [(X_seq[:, :, i], y_seq[i]) for i in 1:size(X_seq, 3)]
    
    # Split data (70% train, 15% val, 15% test)
    n_total = length(data_pairs)
    n_train = Int(floor(0.7 * n_total))
    n_val = Int(floor(0.15 * n_total))
    train_data, val_data, test_data = data_pairs[1:n_train], data_pairs[(n_train+1):(n_train+n_val)], data_pairs[(n_train+n_val+1):end]
    
    println("Train: $(length(train_data)), Val: $(length(val_data)), Test: $(length(test_data))")
    
    # Create DataLoaders
    train_loader = DataLoader(train_data, batchsize=batch_size, shuffle=true)
    val_loader = DataLoader(val_data, batchsize=batch_size, shuffle=false)
    
    # Create and train model
    model = create_lstm_model(size(X, 2), window_size, max_params=max_params)
    trained_params, best_val_acc = train_model(model, train_loader, val_loader)
    
    # Evaluate on test set
    Flux.loadmodel!(model, trained_params)
    test_acc, test_bal_acc = evaluate_model(model, DataLoader(test_data, batchsize=batch_size, shuffle=false))
    println("Final Test Accuracy: $(round(test_acc, digits=4))")
    println("Final Test Balanced Accuracy: $(round(test_bal_acc, digits=4))")
    
    # Save model and preprocessing parameters
    model_filename = "lstm_model_window_$(window_size).jld2"
    JLD2.@save model_filename trained_params trained_st=nothing X_mean X_std feature_cols window_size
    println("Model saved as: $model_filename")
    
    best_models[window_size] = (
        test_bal_acc=test_bal_acc,
        filename=model_filename
    )
end

# Print final results
best_window = argmax([best_models[w].test_bal_acc for w in window_sizes])
best_window_size = window_sizes[best_window]
best_bal_acc = best_models[best_window_size].test_bal_acc

println("FINAL RESULTS")
println("Best window size: $best_window_size")
println("Best balanced accuracy: $(round(best_bal_acc, digits=4))")
println("Best model file: $(best_models[best_window_size].filename)")