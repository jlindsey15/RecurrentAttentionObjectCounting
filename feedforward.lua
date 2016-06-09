


for k = 1,4 do
require 'nn';
net = nn.Sequential()

net:add(nn.SpatialConvolution(3, 12, 10, 10))
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.SpatialConvolution(12,24,10,10))
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2, 2, 2, 2))
net:add(nn.View(24*9*9))
net:add(nn.Linear(24*9*9, 120))
net:add(nn.ReLU())
net:add(nn.Linear(120, 84))
net:add(nn.ReLU())
net:add(nn.Linear(84, 5))
net:add(nn.LogSoftMax())


require 'image'
trainset = {}
testset = {}
traindata = torch.Tensor(4000, 3, 64, 64)
testdata = torch.Tensor(1000, 3, 64, 64)
trainlabel = torch.IntTensor(4000)
testlabel = torch.IntTensor(1000)
for i = 1,5  do
print(i)
for j = 1, 800 do
local img = image.load(('img3/' .. i .. 'cell' .. j .. '.png'), 3, 'byte')
traindata[800*(i-1)+j] = img;
trainlabel[800*(i-1)+j] = i;
end
for j = 801, 1000 do
local img = image.load(('img3/' .. i .. 'cell' .. j .. '.png'), 3, 'byte')
testdata[200*(i-1)+j-800] = img;
testlabel[200*(i-1)+j-800] = i;
end
end

print('a')

trainset.data = traindata
trainset.label = trainlabel
testset.data = testdata
testset.label = testlabel



print(trainset)
print(testset)



setmetatable(trainset,
{__index = function(t, i)
return {t.data[i], t.label[i]}
end}
);



function trainset:size()
return self.data:size(1)
end





mean = {} -- store the mean, to normalize the test set in the future
stdv  = {} -- store the standard-deviation for the future

for i=1,3 do -- over each image channel
mean[i] = trainset.data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
print('Channel ' .. i .. ', Mean: ' .. mean[i])
trainset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction

stdv[i] = trainset.data[{ {}, {i}, {}, {}  }]:std() -- std estimation
print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
trainset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

criterion = nn.ClassNLLCriterion()

print('aaa')
print(k)
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = k


trainer:train(trainset)

for i=1,3 do -- over each image channel
testset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
testset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end


correct = 0
for i = 1,5 do
for j = 1,200 do
local groundtruth = testset.label[5*(i-1)+j]
local prediction = net:forward(testset.data[5*(i-1)+j])
local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
if groundtruth == indices[1] then
correct = correct + 1
end
end
end

print(correct, 100*correct/1000 .. ' % ')

--[[require 'cunn';
net = neta:cuda()
criterion:cuda()
trainset.data = trainset.data:cuda()
trainset.label = trainset.label:cuda()
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 5
trainer:train(trainset)--]]
end

