require 'dp'
require 'rnn'
require 'optim'
require 'torchx'
require 'image'


function loaddata()

trainset = {}
testset = {}
traindata = torch.Tensor(4000, 64, 64, 3)
testdata = torch.Tensor(1000, 64, 64, 3)
trainlabel = torch.IntTensor(4000)
testlabel = torch.IntTensor(1000)
for i = 1, 5 do
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



local nValid = 1000
local nTrain = 4000



print(traindata:size())
local trainInput = dp.ImageView('bhwc', traindata)
local trainTarget = dp.ClassView('b', trainlabel)
local validInput = dp.ImageView('bhwc', testdata)
local validTarget = dp.ClassView('b', testlabel)
local testInput = dp.ImageView('bhwc', testdata)
local testTarget = dp.ClassView('b', testlabel)

trainTarget:setClasses({'1', '2', '3', '4', '5'})
validTarget:setClasses({'1', '2', '3', '4', '5'})
testTarget:setClasses({'1', '2', '3', '4', '5'})


local train = dp.DataSet{inputs=trainInput,targets=trainTarget,which_set='train'}
local valid = dp.DataSet{inputs=validInput,targets=validTarget,which_set='valid'}
local test = dp.DataSet{inputs=validInput,targets=validTarget,which_set='valid'}


local dataset = dp.DataSource{train_set=train,valid_set=valid, test_set=test}
dataset:classes{'1', '2', '3', '4', '5'}
return dataset
end



numHidden1 = 100
numHidden2 = 100
numHidden3 = 200
numHidden4 = 200
numGlimpses = 7
alpha = 0.01
focusSize = 8
numConcentric = 3
betweenWindowFactor = 4


dataset = loaddata()

locationSensor = nn.Sequential()
locationSensor:add(nn.SelectTable(2))
locationSensor:add(nn.Linear(2, numHidden1))
locationSensor:add(nn['ReLU']())

glimpseSensor = nn.Sequential()
glimpseSensor:add(nn.DontCast(nn.SpatialGlimpse(focusSize, numConcentric, betweenWindowFactor):float(),true))
glimpseSensor:add(nn.Collapse(3))
glimpseSensor:add(nn.Linear(dataset:imageSize('c')*(focusSize^2)*numConcentric, numHidden2))
glimpseSensor:add(nn['ReLU']())

glimpse = nn.Sequential()
glimpse:add(nn.ConcatTable():add(locationSensor):add(glimpseSensor))
glimpse:add(nn.JoinTable(1,1))
glimpse:add(nn.Linear(numHidden2+numHidden1, numHidden3))
glimpse:add(nn['ReLU']())
glimpse:add(nn.Linear(numHidden3, numHidden4))


recurrent = nn.Linear(numHidden4, numHidden4)

rnn = nn.Recurrent(numHidden4, glimpse, recurrent, nn['ReLU'](), 10000)

glimpseLoc = nn.Sequential()
glimpseLoc:add(nn.Linear(numHidden4, 2))
glimpseLoc:add(nn.HardTanh())
glimpseLoc:add(nn.MulConstant(20/dataset:imageSize("h")))

attention = nn.RecurrentAttention(rnn, glimpseLoc, numGlimpses, {numHidden4})

agent = nn.Sequential()
agent:add(nn.Convert(dataset:ioShapes(), 'bchw'))
agent:add(attention)

agent:add(nn.SelectTable(-1))
agent:add(nn.Linear(numHidden4, #dataset:classes()))
agent:add(nn.LogSoftMax())

seq = nn.Sequential()
seq:add(nn.Constant(1,1))
seq:add(nn.Add(1))
concat = nn.ConcatTable():add(nn.Identity()):add(seq)
concat2 = nn.ConcatTable():add(nn.Identity()):add(concat)

agent:add(concat2)


for k,param in ipairs(agent:parameters()) do
param:uniform(-0.1, 0.1)
end

learningRate = 0

optimize = dp.Optimizer{
loss = nn.ParallelCriterion(true)
:add(nn.ModuleCriterion(nn.ClassNLLCriterion(), nil, nn.Convert()))
:add(nn.ModuleCriterion(nn.VRClassReward(agent, 1), nil, nn.Convert()))
,
epoch_callback = function(model, report)
if report.epoch > 0 then
alpha = alpha - 0.00001
learningRate = math.max(0.00001, alpha)
end
end,
callback = function(model, report)
model:updateGradParameters(0.9) -- affects gradParams
model:updateParameters(learningRate) -- affects params
model:zeroGradParameters() -- affects gradParams
end,
feedback = dp.Confusion{output_module=nn.SelectTable(1)},
sampler = dp.ShuffleSampler{
epoch_size = -1, batch_size = 20
},
progress = false
}

tester = dp.Evaluator{
feedback = dp.Confusion{output_module=nn.SelectTable(1)},
sampler = dp.Sampler{batch_size = 20}
}


xp = dp.Experiment{
model = agent,
optimizer = optimize,
tester = tester,
max_epoch = 2
}

xp:run(dataset)





for i = 1, 7 do
recurAtt = agent:findModules('nn.RecurrentAttention')[1]
print('aaa')
recurAtt.nStep = i
--[[Experiment]]--
xp = dp.Experiment{
model = agent,
tester = tester,
random_seed = os.time(),
max_epoch = 2
}


xp:run(dataset)

confusionMatrix = xp:tester():feedback()._cm
print(confusionMatrix)

end



