using Flux

W = rand(2, 5)
b = rand(2)

predict(x) = W*x .+ b

function loss(x, y)
    return sum((predict(x) .- y).^2)
end


x, y = rand(5), rand(2)

loss(x,y)

using Flux.Tracker

W = param(W)
b = param(b)

gs = Tracker.gradient(() -> loss(x,y), params(W, b))

using Flux.Tracker: update!

Δ = gs[W]
update!(W, -0.01Δ)

loss(x,y)

# Building Layers

using Flux

W1 = param(rand(3,5))
b1 = param(rand(3))
layer1(x) = W1 * x .+ b1

W2 = param(rand(2,3))
b2 = param(rand(2))

layer2(x) = W2 * x .+ b2

model(x) = layer2(σ.(layer1(x)))

model(rand(5))

function linear(in, out)
    W = param(randn(out,in))
    b = param(randn(out))
    x -> W * x .+ b
end

linear1 = linear(5,3)
linear2 = linear(3,2)
model(x) = linear2(σ.(linear1(x)))

model(rand(5))

# Explicitly create a layer

struct Affine
    W
    b
end

Affine(in::Integer, out::Integer) = Affine(param(randn(out,in)), param(randn(out)))

(m::Affine)(x) = m.W * x .+ m.b

a = Affine(10,5)

a(rand(10))

layer1 = Dense(10, 5, σ)
layer2 = Dense(5, 3, σ)
layer3 = Dense(3,1, σ)

model(x) = layer3(layer2(layer1(x)))

layers = [Dense(10, 10, σ) for i in 1:500000]
model(x) = foldl((x,m) -> m(x), layers, init = x)
@time model(rand(10))

model2 = Chain(
    Dense(10, 5, σ),
    Dense(5,2),
    softmax)

model2(rand(10))
