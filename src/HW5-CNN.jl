using Flux, JLD2, StatsPlots
using MLDatasets:CIFAR10
using CSV, DataFrames, Statistics
using Random


folder = "cifar10"  # sub-directory in which to save
isdir(folder) || mkdir(folder)

test_data  = CIFAR10.testdata(Float32)
train_data = CIFAR10.traindata(Float32)

train_x, train_y = CIFAR10.traindata(Float32)
test_x, test_y = CIFAR10.testdata(Float32)

train_x 
test_y
train_y = Int.(train_y)
test_y = Int.(test_y)

function loader(x, y; batchsize::Int=512, subset::Int=0)
    # if subset specified
    if subset > 0 && subset < size(x, 4)
        indices = shuffle(1:size(x, 4))[1:subset]
        x = x[:, :, :, indices]
        y = y[indices]
    end

    x = mapslices(x -> reverse(permutedims(x ./ 255), dims=1), x, dims=(1, 2))
    yhot = Flux.onehotbatch(Vector(y), 0:9) 
    Flux.DataLoader((x, yhot); batchsize, shuffle=true)
end


function build_lenet(arc_size::Tuple{Int,Int})
    Chain(
        Conv(arc_size, 3 => 6, relu; pad=SamePad()),
        MeanPool((2, 2)),
        Conv(arc_size, 6 => 16, relu; pad=SamePad()),
        MeanPool((2, 2)),
        Flux.flatten,
        Dense(8 * 8 * 16 => 120, relu),
        Dense(120 => 84, relu),
        Dense(84 => 10)
    )
end

using Statistics: mean 

function loss_and_accuracy(model, x, y)
    (x1, y1) = only(loader(x, y; batchsize=size(x, 4))) 
    ŷ = model(x1)
    loss = Flux.logitcrossentropy(ŷ, y1)  # did not include softmax in the model
    acc = round(100 * mean(Flux.onecold(ŷ) .== Flux.onecold(y1)); digits=2)
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
        loss, acc = loss_and_accuracy(model, train_x, train_y)
        test_loss, test_acc = loss_and_accuracy(model, test_x, test_y)
        @info "Epoch: $epoch, Train Acc: $acc, Test Acc: $test_acc"
        @info "Train Loss: $loss, Test Loss: $test_loss"
        push!(train_log, (; epoch, loss, acc, test_loss, test_acc))

        if epoch == epochs
            JLD2.jldsave(filename; model_state=Flux.state(model))
            @info "Saved model to $filename"
        end
    end
    train_log
end


# Task 1
train_data_loader = loader(train_x, train_y; batchsize=32)

# LeNet-5 
lenet5 = build_lenet((5, 5))
train_log5 = train_model!(lenet5, train_data_loader, 6, joinpath(folder, "lenet5.jld2"))


# LeNet-3 
lenet3 = build_lenet((3, 3))
train_log3 = train_model!(lenet3, train_data_loader, 10, joinpath(folder, "lenet3.jld2"))

# LeNet-7 
lenet7 = build_lenet((7, 7))
train_log7 = train_model!(lenet7, train_data_loader, 10, joinpath(folder, "lenet7.jld2"))

filter_accs = [last(train_log3).test_acc, last(train_log5).test_acc, last(train_log7).test_acc]
plot([3, 5, 7], filter_accs, label="Test Accuracy", marker=:circle, xlabel="Filter Size", ylabel="Test Accuracy (%)", title="Effect of Filter Size")
savefig(joinpath(folder, "filter_size_acc.png"))

# Task 3
# 10,000 with 6 epochs
train_loader_10k = loader(train_x, train_y; batchsize=32, subset=10000)
train_log_10k = train_model!(lenet5, train_loader_10k, 6, joinpath(folder, "lenet_10k.jld2"))

# 20,000 with 3 epochs
train_loader_20k = loader(train_x, train_y; batchsize=32, subset=20000)
train_log_20k = train_model!(lenet5, train_loader_20k, 3, joinpath(folder, "lenet_20k.jld2"))

# 30,000 with 2 epochs
train_loader_30k = loader(train_x, train_y; batchsize=32, subset=30000)
train_log_30k = train_model!(lenet5, train_loader_30k, 2, joinpath(folder, "lenet_30k.jld2"))

# Task 4
# Function to visualize feature maps for three samples
function visualize_features(model, samples)
    # Create a plot with 3 rows (one for each sample) and up to 8 columns
    p = plot(layout=(3, 8), size=(1200, 600), titlefont=font(10))
    
    for (i, sample) in enumerate(samples)
        # Preprocess the sample (32×32×3) and add batch dimension (32×32×3×1)
        x = loader(sample[:, :, :, 1:1])
        
        # Plot original image (show first channel for simplicity, as RGB)
        plot!(p[i,1], heatmap(x[:, :, 1, 1], title="Original", c=:grays, clim=(0,1)), 
              aspect_ratio=:equal)
        
        # After first conv layer (Conv1): 32×32×6
        conv1_out = model[1](x)
        for j in 1:min(3, size(conv1_out, 3))  # Show up to 3 channels
            plot!(p[i,1+j], heatmap(conv1_out[:, :, j, 1], title="Conv1 Ch$j", c=:viridis), 
                  aspect_ratio=:equal)
        end
        
        # After first pooling layer (Pool1): 16×16×6
        pool1_out = model[1:2](x)
        for j in 1:min(3, size(pool1_out, 3))  # Show up to 3 channels
            plot!(p[i,4+j], heatmap(pool1_out[:, :, j, 1], title="Pool1 Ch$j", c=:viridis), 
                  aspect_ratio=:equal)
        end
        
        # After second conv layer (Conv2): 16×16×16
        conv2_out = model[1:3](x)
        for j in 1:1  # Show only 1 channel due to large number of channels
            plot!(p[i,7+j], heatmap(conv2_out[:, :, j, 1], title="Conv2 Ch$j", c=:viridis), 
                  aspect_ratio=:equal)
        end
    end
    
    # Save the plot
    savefig(joinpath(folder, "feature_maps_lenet3.png"))
    return p
end

# Select 3 random test samples
sample_indices = shuffle(1:size(test_x, 4))[1:3]
samples = test_x[:, :, :, sample_indices]

# Visualize feature maps
visualize_features(lenet3, samples)

using StatsBase, Plots

function check_balance(labels, dataset_name::String)
    # Count samples per class
    class_counts = countmap(labels)
    
    # Sort by class index (0:9) for consistent display
    sorted_counts = [get(class_counts, i, 0) for i in 0:9]
    
    println("$dataset_name Class Distribution:")
    for (class, count) in enumerate(sorted_counts)
        println("Class $(class-1): $count samples")
    end
end

check_balance(train_y, "Training")
check_balance(test_y, "Testing")