using Flux, JLD2, StatsPlots
using MLDatasets
using CSV, DataFrames, Statistics
using Random

folder = "fashionmnist"
isdir(folder) || mkpath(folder)

train_x, train_y = MLDatasets.FashionMNIST.traindata()
test_x, test_y = MLDatasets.FashionMNIST.testdata()

# Convert labels to integers (they should already be, but make sure)
train_y = Int.(train_y)
test_y = Int.(test_y)

# Check dimensions
@info "Training data size: $(size(train_x))"
@info "Test data size: $(size(test_x))"
@info "Number of training samples: $(size(train_x, 3))"

function loader(x, y; batchsize::Int=512, subset::Int=0)
    
    if subset > 0 && subset < size(x, 3)  
        indices = shuffle(1:size(x, 3))[1:subset]
        x = x[:, :, indices]
        y = y[indices]
    end

    x = Float32.(x) ./ 255.0f0
    x = reshape(x, 28, 28, 1, size(x, 3))  
    yhot = Flux.onehotbatch(y, 0:9)  # One-hot encode labels
    Flux.DataLoader((x, yhot); batchsize, shuffle=true)
end

train_loader = loader(train_x, train_y; batchsize=32)


x1, y1 = first(train_loader)
@show size(x1)  
@show size(y1)  

function build_lenet(kernel_size::Tuple{Int,Int})
    Chain(
        Conv(kernel_size, 1 => 6, relu; pad=SamePad()),
        MeanPool((2, 2)),
        Conv(kernel_size, 6 => 16, relu; pad=SamePad()),
        MeanPool((2, 2)),
        Flux.flatten,
        Dense(7 * 7 * 16 => 120, relu),  
        Dense(120 => 84, relu),
        Dense(84 => 10)
    )
end

using Statistics: mean 

function loss_and_accuracy(model, x, y)
    data_loader = loader(x, y; batchsize=size(x, 3))  
    (x1, y1) = only(data_loader) 
    ŷ = model(x1)
    loss = Flux.logitcrossentropy(ŷ, y1)
    acc = round(100 * mean(Flux.onecold(ŷ) .== Flux.onecold(y1)); digits=2)
    return loss, acc
end

function train_model!(model, train_loader, epochs, filename; eta=0.001, lambda=3e-4)
    opt_rule = AdamW(eta, (0.9, 0.999), lambda)
    opt_state = Flux.setup(opt_rule, model)
    train_log = []
    
    for epoch in 1:epochs
        for (x, y) in train_loader
            grads = Flux.gradient(m -> Flux.logitcrossentropy(m(x), y), model)
            Flux.update!(opt_state, model, grads[1])
        end
        
        train_loss, train_acc = loss_and_accuracy(model, train_x, train_y)
        test_loss, test_acc = loss_and_accuracy(model, test_x, test_y)
        
        @info "Epoch: $epoch, Train Acc: $train_acc%, Test Acc: $test_acc%"
        @info "Train Loss: $train_loss, Test Loss: $test_loss"
        
        push!(train_log, (; epoch, loss=train_loss, acc=train_acc, test_loss, test_acc))

        if epoch == epochs
            JLD2.jldsave(filename; model_state=Flux.state(model))
            @info "Saved model to $filename"
        end
    end
    train_log
end

train_data_loader = loader(train_x, train_y; batchsize=32)

x2, y2 = first(train_data_loader)
@show size(x2)  
@show size(y2)  

# LeNet-5
lenet5 = build_lenet((5, 5))
@info "Model architecture:"
@info lenet5

train_log5 = train_model!(lenet5, train_data_loader, 6, joinpath(folder, "lenet5.jld2"))

using StatsBase, Plots

function check_balance(labels, dataset_name::String)
    # Count samples per class
    class_counts = countmap(labels)
    
    # Sort by class index (0:9) for consistent display
    sorted_counts = [get(class_counts, i, 0) for i in 0:9]
    
    # Print class counts
    println("$dataset_name Class Distribution:")
    for (class, count) in enumerate(sorted_counts)
        println("Class $(class-1): $count samples")
    end
end

check_balance(train_y, "Training")
check_balance(test_y, "Testing")