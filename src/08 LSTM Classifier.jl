using Lux, JLD2, Printf, Random, Statistics, MLUtils, Optimisers, Zygote

function get_dataloaders(; dataset_size=1000, sequence_length=1000)
    # Create the spirals
    data = [MLUtils.Datasets.make_spiral(sequence_length) for _ in 1:dataset_size]
    # Get the labels
    labels = vcat(repeat([0.0f0], dataset_size ÷ 2), repeat([1.0f0], dataset_size ÷ 2))
    clockwise_spirals = [reshape(d[1][:, 1:sequence_length], :, sequence_length, 1)
                         for d in data[1:(dataset_size÷2)]]
    anticlockwise_spirals = [reshape(
        d[1][:, (sequence_length+1):end], :, sequence_length, 1)
                             for d in data[((dataset_size÷2)+1):end]]
    x_data = Float32.(cat(cat(clockwise_spirals..., dims=3), cat(anticlockwise_spirals...; dims=3), dims=3))
    # Split the dataset
    (x_train, y_train), (x_val, y_val) = splitobs((x_data, labels); at=0.8, shuffle=true)
    # Create DataLoaders
    return (
        # Use DataLoader to automatically minibatch and shuffle the data
        DataLoader(collect.((x_train, y_train)); batchsize=128, shuffle=true),
        # Don't shuffle the validation data
        DataLoader(collect.((x_val, y_val)); batchsize=128, shuffle=false))
end

plot(x_data[1, 1:100, 1], x_data[2, 1:100, 1], label="1")
plot!(x_data[1, 101:200, 1], x_data[2, 101:200, 1], label="2")
plot!(x_data[1, 200:300, 1], x_data[2, 200:300, 1], label="3")
plot!(x_data[1, 300:500, 1], x_data[2, 300:500, 1], label="4")
plot!(x_data[1, 500:800, 1], x_data[2, 500:800, 1], label="5")

struct SpiralClassifier{L,C} <: Lux.AbstractLuxContainerLayer{(:lstm_cell, :classifier)}
    lstm_cell::L
    classifier::C
end

function ones_bias(::Xoshiro, n::Int64)
    return ones(Float32, n)
end

function SpiralClassifier(in_dims, hidden_dims, out_dims)
    return Chain(
        Recurrence(LSTMCell(in_dims => hidden_dims)), Dense(hidden_dims => out_dims, sigmoid), vec)
end

const lossfn = BinaryCrossEntropyLoss()

function compute_loss(model, ps, st, (x, y))
    ŷ, st_ = model(x, ps, st)
    loss = lossfn(ŷ, y)
    return loss, st_, (; y_pred=ŷ)
end

matches(y_pred, y_true) = sum((y_pred .> 0.5f0) .== y_true)
accuracy(y_pred, y_true) = matches(y_pred, y_true) / length(y_pred)

# Get the dataloaders
train_loader, val_loader = get_dataloaders()

# Create the model
out = 1
model = SpiralClassifier(2, 2, 1)
rng = Xoshiro(0)
ps, st = Lux.setup(rng, model)

gate(h, n) = (1:h) .+ h * (n - 1)
gate(x::AbstractVector, h, n) = @view x[gate(h, n)]
gate(x::AbstractMatrix, h, n) = view(x, gate(h, n), :)


ps.layer_1.bias_ih[gate(out, 2)] .= 1
ps.layer_1.bias_hh[gate(out, 2)] .= 1


train_state = Training.TrainState(model, ps, st, Adam(0.01f0))

for epoch in 1:5
    # Train the model
    losses = []
    for (x, y) in train_loader
        (_, loss, _, train_state) = Training.single_train_step!(
            AutoZygote(), lossfn, (x, y), train_state)
        append!(losses, loss)
    end
    @printf "Epoch [%3d]: Loss %4.5f\n" epoch sum(losses)

    losses = []
    accs = []
    # Validate the model
    st_ = Lux.testmode(train_state.states)
    for (x, y) in val_loader
        ŷ, st_ = model(x, train_state.parameters, st_)
        loss = lossfn(ŷ, y)
        acc = accuracy(ŷ, y)
        append!(losses, loss)
        append!(accs, acc)
    end
    @printf "Validation: Loss %4.5f Accuracy %4.5f\n" sum(losses) mean(accs)
end
