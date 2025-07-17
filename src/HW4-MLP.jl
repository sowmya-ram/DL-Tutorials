
using Lux, MLUtils, Optimisers, OneHotArrays, Random, Statistics, Printf, Zygote, JLD2, Plots
using CSV, DataFrames

rng = Xoshiro(1)

function flatten(x::AbstractArray)
    return reshape(x, :, size(x)[end])
end

function fashionmnistloader(data::DataFrame, batch_size_)
    x4dim = reshape(permutedims(Matrix{Float32}(select(data, Not(:label)))), 28, 28, 1, :)  # 28x28x1xN
    x4dim = mapslices(x -> reverse(permutedims(x ./ 255), dims=1), x4dim, dims=(1, 2))  # Normalize and flip
    x4dim = meanpool(x4dim, (2, 2))  # Reduce to 14x14
    x4dim = flatten(x4dim)
    yhot = onehotbatch(Vector(data.label), 0:9)  # One-hot encode labels
    return DataLoader((x4dim, yhot); batchsize=batch_size_, shuffle=true)
end


train = CSV.read("./fashionmnist/fashion-mnist_train.csv", DataFrame, header=1)
test = CSV.read("./fashionmnist/fashion-mnist_test.csv", DataFrame, header=1)




train_dataloader_default = fashionmnistloader(train, 512)
test_dataloader_default = fashionmnistloader(test, 10000)


function accuracy(model, ps, st, dataloader)
    total_correct, total = 0, 0
    st = Lux.testmode(st)
    for (x, y) in dataloader
        target_class = onecold(y, 0:9)
        predicted_class = onecold(Array(first(model(x, ps, st))), 0:9)
        total_correct += sum(target_class .== predicted_class)
        total += length(target_class)
    end
    return total_correct / total
end


const lossfn = CrossEntropyLoss(; logits=Val(true))

vjp = AutoZygote()

#Task - 1 Vary hidden layer size 
hidden_sizes = [10, 20, 40, 50, 100, 300]
test_accuracies = Float64[]

for hidden_size in hidden_sizes
    model = Lux.Chain(
        Lux.Dense(196 => hidden_size, relu),
        Lux.Dense(hidden_size => 10)
    )
    ps, st = Lux.setup(rng, model)
    train_state = Training.TrainState(model, ps, st, AdamW(lambda=3e-4))
    
    
    for epoch in 1:10
        for (x, y) in train_dataloader_default
            _, _, _, train_state = Training.single_train_step!(vjp, lossfn, (x, y), train_state)
        end
    end
    
    acc = accuracy(model, train_state.parameters, train_state.states, test_dataloader_default) * 100
    push!(test_accuracies, acc)
    @printf "Hidden size: %d, Test Accuracy: %.2f%%\n" hidden_size acc
end


plot(hidden_sizes, test_accuracies, label="Test Accuracy", xlabel="Hidden Layer Size", ylabel="Accuracy (%)", title="Accuracy vs Hidden Layer Size", marker=:circle)
savefig("./fashionmnist/hidden_size_accuracy.png")


#Task - 2 hidden layer size = 30, random w initialization
hidden_size = 30
test_accuracies_init = Float64[]

model = Lux.Chain(
        Lux.Dense(196 => hidden_size, relu),
        Lux.Dense(hidden_size => 10)
    )

for run in 1:10
    ps, st = Lux.setup(Xoshiro(run), model)  # Different seed for each run
    train_state = Training.TrainState(model, ps, st, AdamW(lambda=3e-4))
    
    for epoch in 1:10
        for (x, y) in train_dataloader_default
            _, _, _, train_state = Training.single_train_step!(vjp, lossfn, (x, y), train_state)
        end
    end
    
    acc = accuracy(model, train_state.parameters, train_state.states, test_dataloader_default) * 100
    push!(test_accuracies_init, acc)
    @printf "Run %d, Test Accuracy: %.2f%%\n" run acc
end

mean_acc = mean(test_accuracies_init)
std_acc = std(test_accuracies_init)
@printf "Mean Test Accuracy: %.2f%%, Standard Deviation: %.2f%%\n" mean_acc std_acc

scatter(1:10, test_accuracies_init, label="Test Accuracy", xlabel="Run", ylabel="Accuracy (%)", title="Test Accuracy Across Initializations")
hline!([mean_acc], label="Mean Accuracy", linestyle=:dash)
savefig("./fashionmnist/random_init_accuracy.png")




#Task -3  batch size of 32 for 25 epochs. Use a decaying learning rate schedule of your choice
model = Lux.Chain(
        Lux.Dense(196 => 30, relu),
        Lux.Dense(30 => 10)
    )

ps, st = Lux.setup(rng, model)
train_dataloader_32 = fashionmnistloader(train, 32)

lr_array = Float64[]
test_accuracies_decay = Float64[]
t3_best_acc = 0.0

for epoch in 1:25
    lr = 3e-4 * (0.5 ^ floor((epoch - 1) / 5))
    opt = AdamW(lambda=lr)
    train_state = Training.TrainState(model, ps, st, opt)
    for (x, y) in train_dataloader_32
        _, _, _, train_state = Training.single_train_step!(vjp, lossfn, (x, y), train_state)
    end
    ps = train_state.parameters
    st = train_state.states
    acc = accuracy(model, ps, st, test_dataloader_default) * 100
    @printf "Epoch %d, LR: %.2e, Test Accuracy: %.2f%%\n" epoch lr acc

    lr_array = vcat(lr_array, lr)
    push!(test_accuracies_decay, acc)

    if acc > t3_best_acc
        t3_best_acc = acc
    end
end
task3_acc = accuracy(model, ps, st, test_dataloader_default) * 100
@printf "Task 3 Final Test Accuracy: %.2f%%\n" task3_acc

@printf "Best Test Accuracy: %.2f%% \n" t3_best_acc
@save "./fashionmnist/task3_model.jld2" ps st



# Task 4: Grid search over batch sizes and learning rate schedules
batch_sizes = [16, 32, 64, 128]
lr_schedules = [(base_lr=3e-4, decay=0.5, every=5), (base_lr=1e-3, decay=0.5, every=5), (base_lr=3e-4, decay=0.7, every=3)]
best_acc = 0.0
best_params = nothing

for batch_size in batch_sizes
    for (base_lr, decay, every) in lr_schedules
        
        train_dataloader = fashionmnistloader(train, batch_size)

        for epoch in 1:10
            lr = base_lr * (decay ^ floor((epoch - 1) / every))
            opt = AdamW(lambda=lr)
            train_state = Training.TrainState(model, ps, st, opt)
            for (x, y) in train_dataloader
                _, _, _, train_state = Training.single_train_step!(vjp, lossfn, (x, y), train_state)
            end
            ps = train_state.parameters
            st = train_state.states
        end
        
        acc = accuracy(model, ps, st, test_dataloader_default) * 100
        @printf "Batch Size: %d, Base LR: %.2e, Decay: %.2f, Every: %d, Test Accuracy: %.2f%%\n" batch_size base_lr decay every acc
        if acc > best_acc
            best_acc = acc
            best_params = (batch_size, base_lr, decay, every)
        end
    end
end

@printf "Best Test Accuracy: %.2f%% with Batch Size: %d, Base LR: %.2e\n" best_acc best_params[1] best_params[2]





train_dataloader = fashionmnistloader(train, best_params[1])
opt = AdamW(lambda=best_params[2])
ps, st = Lux.setup(rng, model)
t5_best_acc = 0.0

for epoch in 1:100
    train_state = Training.TrainState(model, ps, st, opt)
    for (x, y) in train_dataloader
        _, _, _, train_state = Training.single_train_step!(vjp, lossfn, (x, y), train_state)
    end
    ps = train_state.parameters
    st = train_state.states
    acc = accuracy(model, ps, st, test_dataloader_default) * 100
    @printf "Epoch %d, Test Accuracy: %.2f%%\n" epoch  acc

    if acc > t5_best_acc
        t5_best_acc = acc
    end
end
task5_acc = accuracy(model, ps, st, test_dataloader_default) * 100
@printf "Task 5 Final Test Accuracy: %.2f%%\n" task5_acc
@printf "Best Test Accuracy: %.2f%% with Batch Size: %d, Base LR: %.2e\n" t5_best_acc best_params[1] best_params[2]

@printf "Improvement over Task 3: %.2f%% (Task 5: %.2f%%, Task 3: %.2f%%)\n" (task5_acc - task3_acc) task5_acc task3_acc
@save "./fashionmnist/task5_best_model.jld2" ps st
