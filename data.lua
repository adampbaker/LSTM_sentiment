-- Get a random sample of sentence-sentiment pairs, map words to indices, and 
-- organise into batches
-- AB 2015

local stringx = require('pl.stringx')
local file = require('pl.file')
local data_path
local --vocab_map = torch.load(data_path .. "vocab.t7")

function get_document(data_path,docID,max_size)
  local data = file.read(data_path .. docID .. '.csv') -- Read in all the text as a string
  data = stringx.split(data,',') -- Splits data into table, one word per entry
  if docID:sub(1,3) == "pos" then
    label = 2 -- positive review
  else
    label = 1 -- negative review
  end
  local x = {}
  local y = {}
  for i = 1, math.min(#data,max_size) do
    x[i] = data[i]
    y[i] = 0
  end
  y[#y] = label
  return x,y
end

function load_data(state)
  local batch_size = state.batch_size
  local len = state.len

  if state.name == "Training" then
    data_path = "/Users/abaker/smData/IMDB/train/"
  elseif state.name == "Validation" then
    data_path = "/Users/abaker/smData/IMDB/valid/"
  elseif state.name == "Test" then
    data_path = "/Users/abaker/smData/IMDB/test/"
  end

  --if state.data == nil then
    state.data = {}
    state.data.x = torch.ones(len, batch_size)
    state.data.y = torch.zeros(len, batch_size)
  --end
  local x = state.data.x
  local y = state.data.y
  local batch_idx = 1
  local input, target
  local sentiment = "neg"

  for batch_idx = 1,batch_size do
    if sentiment == "neg" then
      sentiment = "pos"
      state.case = state.case + 1
    else
      sentiment = "neg"
    end
    local docID = sentiment .. state.cases[state.case]
    input, target = get_document(data_path,docID,len)
    for j = 1, #input do
      x[j][batch_idx] = input[j]
      y[j][batch_idx] = target[j]
    end
  end
end
