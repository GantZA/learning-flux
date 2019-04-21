using Flux
using Flux.Data.MNIST
using Flux: @epochs, back!, testmode!, throttle
using Base.Iterators: partition
using Distributions: Uniform
using Statistics: mean
using CUDAnative: tanh, log, exp
using CuArrays

BATCH_SIZE = 128
imgs = MNIST.images()

data = [float(hcat(vec.(imgs)...)) for imgs in partition(imgs, BATCH_SIZE)]

NUM_EPOCHS = 5
noise_dim = 96
channels = 128
hidden_dim = 7 * 7 * channels

dist = Uniform(-1,1)
training_steps = 0
verbose_freq = 100

############################### DCGAN Architecture #############################
################################## Generator ###################################

fc_gen = Chain(
    Dense(noise_dim, 1024, relu),
    BatchNorm(1024),
    Dense(1024, hidden_dim, relu),
    BatchNorm(hidden_dim))

deconv = Chain(
    ConvTranspose((4,4), channels=>64, relu; stride=(2,2), pad=(1,1)),
    BatchNorm(64),
    ConvTranspose((4,4), 64=>1, tanh; stride=(2,2), pad=(1,1)))

generator = Chain(
    fc_gen..., x->reshape(x, 7, 7, channels, :),
    deconv..., x->reshape(x, 784,:)) |> gpu

################################## Discriminator ###############################
fc_disc = Chain(
    Dense(1024, 1024, leakyrelu),
    Dense(1024,1))

conv = Chain(
    Conv((5,5), 1=>32, leakyrelu),
    x -> maxpool(x, (2,2)),
    Conv((5,5), 32=>64, leakyrelu),
    x -> maxpool(x, (2,2)))

discriminator = Chain(
    x->reshape(x,28,28,1,:),
    conv..., x -> reshape(x, 1024, :), fc_disc...) |> gpu

###############################################################################

opt_gen = ADAM(params(generator), 0.001f0, β1 = 0.5)
opt_disc = ADAM(params(discriminator), 0.001f0, β1 = 0.5)

##################### helper functions ########################################

function nullify_grad!(p)
    if typeof(p) <: TrackedArray
        p.grad .= 0.0f0
    end
    return p
end

# Convert internal parameters for defined model so that they become Tracked cuArray
function zero_grad!(model)
    model = mapleaves(nullify_grad!, model)
end

using Images

img(x) = Gray.(reshape((x+1)/2, 28, 28))

function sample()
    noise = [rand(dist, noise_dim, 1) for i in 1:36]

    # Generating Images
    testmode!(generator)
    fake_imgs = img.(map(x -> cpu(generator(x).data), noise))
    testmode!(generator, false)

    img_grid = vcat([hcat(imgs...) for imgs in partition(fake_imgs, 6)]...)
end

# Loss and training

function bce(yhat, y)
    neg_abs = -abs.(yhat)
    mean(relu.(yhat) .- yhat .* y .+ log.(1 .+ exp.(neg_abs)))
end

function train(x)
    global training_steps

    z = rand(dist, noise_dim, BATCH_SIZE) |> gpu
    inp = 2x .- 1 |> gpu

    #zero_grad!(discriminator)

    D_real = discriminator(inp)
    D_real_loss = bce(D_real, ones(Float32,size(D_real.data)))

    fake_x = generator(z)
    D_fake = discriminator(fake_x)
    D_fake_loss = bce(D_fake, zero(D_fake.data))

    D_loss = D_fake_loss + D_real_loss

    Flux.back!(D_loss)
    opt_disc()

    z = rand(dist, noise_dim, BATCH_SIZE)

    fake_x = generator(z)
    D_fake = discriminator(fake_x)
    G_loss = bce(D_fake, ones(Float32,size(D_fake.data)))

    Flux.back!(G_loss)
    opt_gen()

    if training_steps % verbose_freq == 0
        println("D Loss: $(D_loss.data) | G loss: $(G_loss.data)")
    end
    training_steps += 1
    param(0.0f0)

end

for i = 1:NUM_EPOCHS
  for imgs in data
    train(imgs)
  end
  println("Epoch $i over.")
end


z = rand(dist, noise_dim, BATCH_SIZE) |> cpu
inp = 2data[1] .- 1 |> cpu

#zero_grad!(discriminator)

D_real = discriminator(inp)
D_real_loss = bce(D_real.data, ones(size(D_real.data)))

yhat = D_real.data
y = ones(Float32,size(D_real.data))
neg_abs = -abs.(yhat)
mean(relu.(yhat) .- yhat .* y .+ log.(1 + exp.(neg_abs)))

log.(1 + exp.(neg_abs))

save("sample_dcgan.png", sample())







a = 0.0f0

typeof(a)


z = rand(dist, noise_dim, BATCH_SIZE) |> cpu
inp = 2imgs[1] .- 1 |> cpu

#zero_grad!(discriminator)

D_real = discriminator(inp)
@time ones(size(D_real.data))

using Juno
