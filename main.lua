include "data.lua"
require "utils/utils"
require "layers/MaskedLoss.lua"
require "layers/Embedding.lua"
require "csvigo"

function lstm(i, prev_c, prev_h)
  function new_input_sum()
    local i2h            = nn.Linear(params.rnn_size, params.rnn_size)
    local h2h            = nn.Linear(params.rnn_size, params.rnn_size)
    return nn.CAddTable()({i2h(i), h2h(prev_h)})
  end
  local in_gate          = nn.Sigmoid()(new_input_sum())
  local forget_gate      = nn.Sigmoid()(new_input_sum())
  local in_gate2         = nn.Tanh()(new_input_sum())
  local next_c           = nn.CAddTable()({
    nn.CMulTable()({forget_gate, prev_c}),
    nn.CMulTable()({in_gate,     in_gate2})
  })
  local out_gate         = nn.Sigmoid()(new_input_sum())
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
  return next_c, next_h
end

function create_network()
  local x                = nn.Identity()()
  local y                = nn.Identity()()
  local prev_s           = nn.Identity()()
  local i                = {[0] = Embedding(params.vocab_size,
                                            params.rnn_size)(x)}
  local next_s           = {}
  local splitted         = {prev_s:split(2 * params.layers)}
  for layer_idx = 1, params.layers do
    local prev_c         = splitted[2 * layer_idx - 1]
    local prev_h         = splitted[2 * layer_idx]
    local dropped        = nn.Dropout(params.dropout)(i[layer_idx - 1])
    local next_c, next_h = lstm(dropped, prev_c, prev_h)
    table.insert(next_s, next_c)
    table.insert(next_s, next_h)
    i[layer_idx] = next_h
  end
  local h2y              = nn.Linear(params.rnn_size, 2)
  local pred             = nn.LogSoftMax()(h2y(i[params.layers]))
  local err              = MaskedLoss()({pred, y})
  local module           = nn.gModule({x, y, prev_s}, 
                                      {err, nn.Identity()(next_s)})
  module:getParameters():uniform(-params.init_weight, params.init_weight)
  return module:cuda()
end

function setup()
  print("Creating a RNN LSTM network.")
  local core_network = create_network()
  paramx, paramdx = core_network:getParameters()
  model = {}
  model.s = {}
  model.ds = {}
  model.start_s = {}
  for j = 0, params.seq_length do
    model.s[j] = {}
    for d = 1, 2 * params.layers do
      model.s[j][d] = torch.zeros(params.batch_size, params.rnn_size):cuda()
    end
  end
  for d = 1, 2 * params.layers do
    model.start_s[d] = torch.zeros(params.batch_size, params.rnn_size):cuda()
    model.ds[d] = torch.zeros(params.batch_size, params.rnn_size):cuda()
  end
  model.core_network = core_network
  model.rnns = cloneManyTimes(core_network, params.seq_length)
  model.norm_dw = 0
  reset_ds()
end

function reset_state(state)
  --load_data(state)
  state.pos = 1
  if model ~= nil and model.start_s ~= nil then
    for d = 1, 2 * params.layers do
      model.start_s[d]:zero()
    end
  end
end

function reset_stats(state)
  state.acc = 0
  state.count = 0
  state.normal = 0
  state.predictions = {}
end

function reset_ds()
  for d = 1, #model.ds do
    model.ds[d]:zero()
  end
end

function fp(state, paramx_)
  if paramx_ ~= paramx then paramx:copy(paramx_) end
  copy_table(model.s[0], model.start_s)
  if state.pos + params.seq_length > state.data.x:size(1) then
    reset_state(state)
  end
  --print(state.data.y:sum(1))
  local pred = torch.zeros(state.batch_size)
  for i = 1, params.seq_length do
    tmp, model.s[i] = unpack(model.rnns[i]:forward({state.data.x[state.pos],
                                                    state.data.y[state.pos],
                                                    model.s[i - 1]}))
    cutorch.synchronize()
    state.pos = state.pos + 1
    state.count = state.count + tmp[2]
    state.normal = state.normal + tmp[3]
    pred = pred + tmp[4]
  end
  for i = 1, pred:size(1) do
    state.predictions[#state.predictions+1] = pred[i]
  end
  state.acc = state.count / state.normal
  copy_table(model.start_s, model.s[params.seq_length])
end

function bp(state)
  paramdx:zero()
  reset_ds()
  for i = params.seq_length, 1, -1 do
    state.pos = state.pos - 1
    local tmp = model.rnns[i]:backward({state.data.x[state.pos],
                                        state.data.y[state.pos],
                                        model.s[i - 1]},
                                       {torch.ones(1):cuda(), model.ds})[3]
    copy_table(model.ds, tmp)
    cutorch.synchronize()
  end
  state.pos = state.pos + params.seq_length
  model.norm_dw = paramdx:norm()
  if model.norm_dw > params.max_grad_norm then
    shrink_factor = params.max_grad_norm / model.norm_dw
    paramdx:mul(shrink_factor)
  end
end

function eval_training(paramx_)
  fp(state_train, paramx_)
  bp(state_train)
  return 0, paramdx
end



function save_predictions(state)
  csvfile = "./results/" .. state.name .. "_predictions.csv"
  file = assert(io.open(csvfile, "w"))
  for i=1,#state.predictions do
    file:write(state.predictions[i])
    file:write('\n')
  end
  file:close()
end


function run_test(state)
  reset_state(state)
  reset_stats(state)
  state.case = 0
  while state.case + state.batch_size/2 <= state.total_cases do
    load_data(state)
    fp(state, paramx)
  end
  print(state.name .. " acc = " .. state.acc)
  --save_predictions(state)
  --print(state.predictions)
end





function main()
  cmd = torch.CmdLine()
  cmd:option('-gpuidx', 1, 'Index of GPU on which job should be executed.')
  cmd:text()
  opt = cmd:parse(arg)

  init_gpu(opt.gpuidx)
  params =      {batch_size=20,
                 seq_length=300,
                 layers=1,
                 rnn_size=64,
                 dropout=0.5,
                 vocab_size=10000,
                 init_weight=0.08,
                 learningRate=0.01,
                 max_grad_norm=5,
                target_accuracy=0.95}
  state_train = {len=params.seq_length,
                 case=0,
                 total_cases=10000,
                 batch_size=params.batch_size,
                 name="Training"}
  state_valid =  {len=params.seq_length,
                 case=0,
                 total_cases=2500,
                 batch_size=params.batch_size,
                 name="Validation"}
  state_test =  {len=params.seq_length,
                 case=0,
                 total_cases = 12500,
                 batch_size=params.batch_size,
                 name="Test"}
  print("Network parameters:")
  print(params)
  states = {state_train, state_valid, state_test}
  for _, state in pairs(states) do
    reset_state(state)
    reset_stats(state)
    state.case = 0
    if state.name == "Training" then
      state.cases = torch.randperm(state.total_cases)
    else
      state.cases = torch.range(1,state.total_cases)
    end
  end
  setup()
  step = 0
  epoch = 0
  max_epochs = 100
  train_accs = {}
  cases_processed = 0
  start_time = torch.tic()
  print("Starting training.")


while true do
  step = step + 1

  if state_train.case + state_train.batch_size/2 > state_train.total_cases then
    print("epoch=" .. epoch)
    reset_state(state_train)
    reset_stats(state_train)
    state_train.cases = torch.randperm(state_train.total_cases)
    state_train.case = 0
    epoch = epoch + 1
    run_test(state_test)
    if epoch > max_epochs then
      break
    end
  end

  load_data(state_train)
  optim.adagrad(eval_training, paramx, {learningRate=params.learningRate}, {})
  cases_processed = cases_processed + state_train.batch_size*state_train.len

  if state_train.case > 0 and state_train.case % 200 == 0 then
    params.learningRate = params.learningRate * 0.8
    print("Training acc = " .. state_train.acc)
    run_test(state_valid)
    reset_state(state_train)
    reset_stats(state_train)
    ----------------------------------------------------------------------------
    wps = floor(cases_processed / torch.toc(start_time))
    print("Words per second = " .. wps)
    cases_processed = 0
    start_time = torch.tic()
    ----------------------------------------------------------------------------
  end

  if step % 33 == 0 then
    collectgarbage()
  end 

end

end

main()
