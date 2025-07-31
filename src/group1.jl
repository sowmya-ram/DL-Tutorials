using CSV
using DataFrames
using Flux
using Flux: onehotbatch, onecold
using Statistics
using JLD2
using Random

# Set random seed for reproducibility
Random.seed!(42)

# Load and preprocess data
df = CSV.read("./datasets/Group A Dataset.csv", DataFrame, header=1)

# Drop fnlwgt and education columns
select!(df, Not([:fnlwgt, :education]))

# Replace "?" with "Unknown" for categorical columns
for col in [:workclass, :occupation, :native_country]
    df[!, col] = replace(df[!, col], "?" => "Unknown")
end

# Define features and target
features = [:age, :workclass, :education_num, :marital_status, :occupation, 
            :relationship, :race, :sex, :capital_gain, :capital_loss, 
            :hour_per_week, :native_country]
target = :label

# Encode categorical variables
categorical_cols = [:workclass, :marital_status, :occupation, :relationship, 
                    :race, :sex, :native_country]
cat_mappings = Dict()
for col in categorical_cols
    unique_vals = unique(df[!, col])
    cat_mappings[col] = Dict(val => i for (i, val) in enumerate(unique_vals))
end

# Convert categorical columns to indices
for col in categorical_cols
    df[!, col] = [cat_mappings[col][val] for val in df[!, col]]
end

# Encode labels (<=50K: 0, >50K: 1)
label_mapping = Dict("<=50K" => 0, "<=50K." => 0, ">50K" => 1, ">50K." => 1)
df[!, :label] = [label_mapping[val] for val in df[!, :label]]

# Normalize numerical features
numerical_cols = [:age, :education_num, :capital_gain, :capital_loss, :hour_per_week]
norm_params = Dict()
for col in numerical_cols
    μ = mean(df[!, col])
    σ = std(df[!, col])
    df[!, col] = (df[!, col] .- μ) ./ (σ + 1e-8)
    norm_params[col] = (mean=μ, std=σ)
end

# Split features and target
X = Matrix{Float32}(df[!, features])
y = df[!, :label]

# Split into train and validation (80-20 split)
n = size(X, 1)
indices = shuffle(1:n)
train_idx = indices[1:floor(Int, 0.8*n)]
val_idx = indices[floor(Int, 0.8*n)+1:end]

X_train = X[train_idx, :]
y_train = y[train_idx]
X_val = X[val_idx, :]
y_val = y[val_idx]

# Create data loaders
train_data = Flux.DataLoader((X_train', y_train), batchsize=128, shuffle=true)

# Define a simple neural network (to stay under 1000 parameters)
input_dim = size(X_train, 2)  # 12 features
hidden_dim = 20
output_dim = 1

model = Chain(
    Dense(input_dim, hidden_dim, relu),
    Dense(hidden_dim, output_dim, sigmoid)
)

# Calculate number of parameters
n_params = sum(length, Flux.params(model))
println("Number of parameters: $n_params")  # Should be < 1000

# Define loss function (binary cross-entropy)
loss(x, y) = Flux.binarycrossentropy(model(x), y)

# Setup optimizer (AdamW with specified parameters)
opt = AdamW(0.001, (0.9, 0.999), 0.001)

# Training loop
ps = Flux.params(model)
for epoch in 1:100
    for (x, y) in train_data
        gs = gradient(() -> loss(x, y), ps)
        Flux.update!(opt, ps, gs)
    end
    # Compute validation accuracy
    y_pred = model(X_val') .> 0.5
    bal_acc = mean((y_pred .== y_val) .* (y_val .== 1)) / mean(y_val .== 1) +
              mean((y_pred .== y_val) .* (y_val .== 0)) / mean(y_val .== 0)
    bal_acc /= 2
    println("Epoch $epoch, Validation Balanced Accuracy: $bal_acc")
end

# Save model and preprocessing parameters
save_dict = Dict(
    "model" => model,
    "cat_mappings" => cat_mappings,
    "norm_params" => norm_params,
    "label_mapping" => label_mapping
)
JLD2.save("model.jld2", save_dict)